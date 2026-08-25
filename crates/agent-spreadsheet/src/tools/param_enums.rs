use schemars::JsonSchema;
use serde::de;
use serde::{Deserialize, Serialize};
use std::fmt;

fn normalize_literal(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn levenshtein_distance(left: &str, right: &str) -> usize {
    if left.is_empty() {
        return right.chars().count();
    }
    if right.is_empty() {
        return left.chars().count();
    }

    let right_chars: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right_chars.len()).collect();
    let mut current = vec![0; right_chars.len() + 1];

    for (i, left_ch) in left.chars().enumerate() {
        current[0] = i + 1;
        for (j, right_ch) in right_chars.iter().enumerate() {
            let substitution_cost = if left_ch == *right_ch { 0 } else { 1 };
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + substitution_cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right_chars.len()]
}

fn suggest_literal<'a>(input: &str, valid: &'a [&'a str]) -> Option<&'a str> {
    let normalized_input = normalize_literal(input);
    let mut best: Option<(&str, usize)> = None;

    for candidate in valid {
        let distance = levenshtein_distance(&normalized_input, &normalize_literal(candidate));
        match best {
            Some((_, best_distance)) if distance >= best_distance => {}
            _ => best = Some((candidate, distance)),
        }
    }

    match best {
        Some((candidate, distance)) if distance <= 6 => Some(candidate),
        _ => None,
    }
}

fn enum_value_error(label: &str, input: &str, valid: &[&str], suggestion: Option<&str>) -> String {
    let valid_list = valid.join("|");
    match suggestion {
        Some(candidate) if !candidate.eq_ignore_ascii_case(input) => {
            format!("invalid {label} '{input}'. Did you mean '{candidate}'? valid: {valid_list}")
        }
        _ => format!("invalid {label} '{input}'. valid: {valid_list}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum BatchMode {
    #[default]
    Apply,
    Preview,
}

impl BatchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Apply => "apply",
            Self::Preview => "preview",
        }
    }

    pub fn is_preview(self) -> bool {
        matches!(self, Self::Preview)
    }
}

impl fmt::Display for BatchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for BatchMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_ascii_lowercase().as_str() {
            "apply" => Ok(Self::Apply),
            "preview" => Ok(Self::Preview),
            other => {
                let valid = ["apply", "preview"];
                let message =
                    enum_value_error("batch_mode", other, &valid, suggest_literal(other, &valid));
                Err(de::Error::custom(message))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ReplaceMatchMode {
    #[default]
    Exact,
    Contains,
}

impl ReplaceMatchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Contains => "contains",
        }
    }
}

impl<'de> Deserialize<'de> for ReplaceMatchMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_ascii_lowercase().as_str() {
            "exact" => Ok(Self::Exact),
            "contains" => Ok(Self::Contains),
            other => {
                let valid = ["exact", "contains"];
                let message =
                    enum_value_error("match_mode", other, &valid, suggest_literal(other, &valid));
                Err(de::Error::custom(message))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FillDirection {
    Down,
    Right,
    #[default]
    Both,
}

impl FillDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Right => "right",
            Self::Both => "both",
        }
    }
}

impl<'de> Deserialize<'de> for FillDirection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_ascii_lowercase().as_str() {
            "down" => Ok(Self::Down),
            "right" => Ok(Self::Right),
            "both" => Ok(Self::Both),
            other => {
                let valid = ["down", "right", "both"];
                let message = enum_value_error(
                    "fill_direction",
                    other,
                    &valid,
                    suggest_literal(other, &valid),
                );
                Err(de::Error::custom(message))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FormulaRelativeMode {
    #[default]
    Excel,
    AbsCols,
    AbsRows,
}

impl<'de> Deserialize<'de> for FormulaRelativeMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_ascii_lowercase().as_str() {
            "excel" => Ok(Self::Excel),
            "abs_cols" | "abscols" | "columns_absolute" => Ok(Self::AbsCols),
            "abs_rows" | "absrows" | "rows_absolute" => Ok(Self::AbsRows),
            other => {
                let valid = ["excel", "abs_cols", "abs_rows"];
                let direct_suggestion = match other {
                    "fully_relative" | "fullyrelative" => Some("excel"),
                    _ => None,
                };
                let message = enum_value_error(
                    "relative_mode",
                    other,
                    &valid,
                    direct_suggestion.or_else(|| suggest_literal(other, &valid)),
                );
                Err(de::Error::custom(message))
            }
        }
    }
}

impl From<FormulaRelativeMode> for crate::formula::pattern::RelativeMode {
    fn from(value: FormulaRelativeMode) -> Self {
        match value {
            FormulaRelativeMode::Excel => Self::Excel,
            FormulaRelativeMode::AbsCols => Self::AbsCols,
            FormulaRelativeMode::AbsRows => Self::AbsRows,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PageOrientation {
    Portrait,
    Landscape,
}

impl<'de> Deserialize<'de> for PageOrientation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_ascii_lowercase().as_str() {
            "portrait" => Ok(Self::Portrait),
            "landscape" => Ok(Self::Landscape),
            other => {
                let valid = ["portrait", "landscape"];
                let message = enum_value_error(
                    "page_orientation",
                    other,
                    &valid,
                    suggest_literal(other, &valid),
                );
                Err(de::Error::custom(message))
            }
        }
    }
}

impl PageOrientation {
    pub fn to_umya(self) -> umya_spreadsheet::OrientationValues {
        match self {
            Self::Portrait => umya_spreadsheet::OrientationValues::Portrait,
            Self::Landscape => umya_spreadsheet::OrientationValues::Landscape,
        }
    }
}
