use crate::model::FormulaGroup;
use crate::utils::column_number_to_name;
use anyhow::{Context, Result};
use formualizer_parse::{
    ASTNode,
    parser::{BatchParser, ReferenceType},
    pretty::canonical_formula,
};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use umya_spreadsheet::{CellFormulaValues, Worksheet};

#[derive(Clone)]
pub struct FormulaAtlas {
    parser: Arc<Mutex<BatchParser>>,
    cache: Arc<RwLock<HashMap<String, Arc<ParsedFormula>>>>,
    _volatility: Arc<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ParsedFormula {
    pub fingerprint: String,
    pub canonical: String,
    pub is_volatile: bool,
    pub dependencies: Vec<String>,
}

impl FormulaAtlas {
    pub fn new(volatility_functions: Vec<String>) -> Self {
        let normalized: Vec<String> = volatility_functions
            .into_iter()
            .map(|s| s.to_ascii_uppercase())
            .collect();
        let lookup = Arc::new(normalized);
        let closure_lookup = lookup.clone();
        let parser = BatchParser::builder()
            .with_volatility_classifier(move |name| {
                let upper = name.to_ascii_uppercase();
                closure_lookup.iter().any(|entry| entry == &upper)
            })
            .build();
        Self {
            parser: Arc::new(Mutex::new(parser)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            _volatility: lookup,
        }
    }

    pub fn parse(&self, formula: &str) -> Result<Arc<ParsedFormula>> {
        if let Some(existing) = self.cache.read().get(formula) {
            return Ok(existing.clone());
        }

        let ast = {
            let mut parser = self.parser.lock();
            parser
                .parse(formula)
                .with_context(|| format!("failed to parse formula: {formula}"))?
        };
        let parsed = Arc::new(parsed_from_ast(&ast));

        self.cache
            .write()
            .insert(formula.to_string(), parsed.clone());
        Ok(parsed)
    }
}

impl Default for FormulaAtlas {
    fn default() -> Self {
        Self::new(default_volatility_functions())
    }
}

fn parsed_from_ast(ast: &ASTNode) -> ParsedFormula {
    let fingerprint = format!("{:016x}", ast.fingerprint());
    let canonical = canonical_formula(ast);
    let dependencies = ast
        .get_dependencies()
        .iter()
        .map(|reference| reference_to_string(reference))
        .collect();
    ParsedFormula {
        fingerprint,
        canonical,
        is_volatile: ast.contains_volatile(),
        dependencies,
    }
}

pub struct FormulaGraph {
    precedents: HashMap<String, Vec<String>>,
    dependents: HashMap<String, Vec<String>>,
    groups: HashMap<String, FormulaGroupAccumulator>,
}

impl FormulaGraph {
    pub fn build(sheet: &Worksheet, atlas: &FormulaAtlas) -> Result<Self> {
        let mut precedents: HashMap<String, Vec<String>> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
        let mut groups: HashMap<String, FormulaGroupAccumulator> = HashMap::new();

        for cell in sheet.get_cell_collection() {
            if !cell.is_formula() {
                continue;
            }
            let formula_text = cell.get_formula();
            if formula_text.is_empty() {
                continue;
            }
            let parsed = atlas.parse(formula_text)?;
            let coordinate = cell.get_coordinate();
            let address = coordinate.get_coordinate();

            let (is_array, is_shared_type) = cell
                .get_formula_obj()
                .map(|obj| match obj.get_formula_type() {
                    CellFormulaValues::Array => (true, false),
                    CellFormulaValues::Shared => (false, true),
                    _ => (false, false),
                })
                .unwrap_or((false, false));

            let group = groups.entry(parsed.fingerprint.clone()).or_insert_with(|| {
                FormulaGroupAccumulator {
                    canonical: parsed.canonical.clone(),
                    addresses: Vec::new(),
                    is_volatile: parsed.is_volatile,
                    is_array,
                    is_shared: is_shared_type,
                }
            });
            if cell.get_formula_shared_index().is_some() {
                group.is_shared = true;
            }
            group.addresses.push(address.clone());
            group.is_volatile |= parsed.is_volatile;

            for dependency in &parsed.dependencies {
                precedents
                    .entry(address.clone())
                    .or_default()
                    .push(dependency.clone());
                dependents
                    .entry(dependency.clone())
                    .or_default()
                    .push(address.clone());
            }
        }

        Ok(Self {
            precedents,
            dependents,
            groups,
        })
    }

    pub fn groups(&self) -> Vec<FormulaGroup> {
        self.groups
            .iter()
            .map(|(fingerprint, group)| FormulaGroup {
                fingerprint: fingerprint.clone(),
                addresses: group.addresses.clone(),
                formula: group.canonical.clone(),
                is_array: group.is_array,
                is_shared: group.is_shared,
                is_volatile: group.is_volatile,
            })
            .collect()
    }

    pub fn precedents(&self, address: &str) -> Vec<String> {
        self.precedents.get(address).cloned().unwrap_or_default()
    }

    pub fn dependents(&self, address: &str) -> Vec<String> {
        self.dependents.get(address).cloned().unwrap_or_default()
    }
}

struct FormulaGroupAccumulator {
    canonical: String,
    addresses: Vec<String>,
    is_volatile: bool,
    is_array: bool,
    is_shared: bool,
}

fn reference_to_string(reference: &ReferenceType) -> String {
    reference.to_string()
}

pub fn normalize_cell_reference(sheet_name: &str, row: u32, col: u32) -> String {
    format!("{}!{}{}", sheet_name, column_number_to_name(col), row)
}

fn default_volatility_functions() -> Vec<String> {
    vec![
        "NOW",
        "TODAY",
        "RAND",
        "RANDBETWEEN",
        "OFFSET",
        "INDIRECT",
        "INFO",
        "CELL",
        "AREAS",
        "INDEX",
        "MOD",
        "ROW",
        "COLUMN",
        "ROWS",
        "COLUMNS",
        "HYPERLINK",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}
