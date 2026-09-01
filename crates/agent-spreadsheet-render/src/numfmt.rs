//! A bounded number formatter.
//!
//! Why this exists rather than umya's `Cell::get_formatted_value`: the 5003
//! bake-off measured umya mis-rendering multi-section formats. Under
//! `0.0%;[Red]-0.0%` it returned `57.0%` for 0.5727 (Excel, LibreOffice,
//! BetterOffice and readany-render all give `57.3%`) and `3.0%` for -0.032
//! (correct is `-3.2%`, in red). Both the rounding and the sign were wrong,
//! and that one defect accounted for 22 of 45 corpus workbooks raising
//! `number_format_approximated`.
//!
//! What is supported, exactly:
//!
//! * section splitting — `positive;negative;zero;text`
//! * `[Red]` and the other seven named colours, and `[Color n]`
//! * the digit placeholders `0`, `#` and `?`, the decimal point, the thousands
//!   comma, and `%`
//! * scientific notation (`0.00E+00`)
//! * quoted literals, backslash escapes, `_x` width reserves, `*x` fills
//! * the `@` text placeholder
//! * `General`
//!
//! Everything else — dates and times, conditional sections (`[>100]`), locale
//! and currency codes (`[$-409]`), elapsed time (`[h]`), fractions (`# ?/?`),
//! and trailing comma scaling — returns `None`. The caller then falls back to
//! umya and raises `number_format_approximated`, so an approximation is always
//! declared.

use crate::scene::Rgba;

/// A formatted cell value: the text, and the colour the format section forces
/// (`None` leaves the cell's own font colour alone).
#[derive(Debug, Clone, PartialEq)]
pub struct Formatted {
    pub text: String,
    pub color: Option<Rgba>,
}

/// The 56-entry legacy indexed palette, used by `[Color n]`. Same table Excel
/// and umya carry; `[Color n]` is 1-based into it.
const INDEXED_COLORS: [&str; 56] = [
    "FF000000", "FFFFFFFF", "FFFF0000", "FF00FF00", "FF0000FF", "FFFFFF00", "FFFF00FF", "FF00FFFF",
    "FF000000", "FFFFFFFF", "FFFF0000", "FF00FF00", "FF0000FF", "FFFFFF00", "FFFF00FF", "FF00FFFF",
    "FF800000", "FF008000", "FF000080", "FF808000", "FF800080", "FF008080", "FFC0C0C0", "FF808080",
    "FF9999FF", "FF993366", "FFFFFFCC", "FFCCFFFF", "FF660066", "FFFF8080", "FF0066CC", "FFCCCCFF",
    "FF000080", "FFFF00FF", "FFFFFF00", "FF00FFFF", "FF800080", "FF800000", "FF008080", "FF0000FF",
    "FF00CCFF", "FFCCFFFF", "FFCCFFCC", "FFFFFF99", "FF99CCFF", "FFFF99CC", "FFCC99FF", "FFFFCC99",
    "FF3366FF", "FF33CCCC", "FF99CC00", "FFFFCC00", "FFFF9900", "FFFF6600", "FF666699", "FF969696",
];

/// Values at or beyond this magnitude are out of the formatter's bounded
/// range: the decimal round trip stops being exact and Excel switches to its
/// own overflow behaviour. They fall back.
const MAX_ABS: f64 = 1e15;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Token {
    Zero,
    Hash,
    Question,
    Point,
    Comma,
    Percent,
    /// `E+` or `E-`.
    Exponent {
        explicit_plus: bool,
    },
    Text,
    /// A literal character emitted verbatim.
    Literal(char),
}

/// Format `value` under `code`. `None` means "this format is outside the
/// bounded grammar" and the caller must fall back.
pub fn format_number(value: f64, code: &str) -> Option<Formatted> {
    if !value.is_finite() || value.abs() >= MAX_ABS {
        return None;
    }
    let sections = split_sections(code)?;
    // Excel picks the section by sign, then formats the magnitude: a negative
    // section supplies its own sign, which is precisely what umya loses.
    let (section, force_negative_sign) = match sections.len() {
        1 => (sections[0].as_str(), value < 0.0),
        2 => {
            if value < 0.0 {
                (sections[1].as_str(), false)
            } else {
                (sections[0].as_str(), false)
            }
        }
        _ => {
            if value > 0.0 {
                (sections[0].as_str(), false)
            } else if value < 0.0 {
                (sections[1].as_str(), false)
            } else {
                (sections[2].as_str(), false)
            }
        }
    };
    render_section(section, value.abs(), force_negative_sign)
}

/// Format a string cell under `code`. Only the fourth (text) section is
/// consulted; a format with no text section leaves the string untouched.
pub fn format_text(text: &str, code: &str) -> Option<Formatted> {
    let sections = split_sections(code)?;
    let Some(section) = sections.get(3) else {
        return Some(Formatted {
            text: text.to_string(),
            color: None,
        });
    };
    let (color, body) = take_color(section)?;
    let tokens = tokenize(&body)?;
    let mut out = String::new();
    for token in &tokens {
        match token {
            Token::Text => out.push_str(text),
            Token::Literal(c) => out.push(*c),
            _ => return None,
        }
    }
    Some(Formatted { text: out, color })
}

/// Split on `;` at bracket/quote depth zero. Excel allows at most four
/// sections.
fn split_sections(code: &str) -> Option<Vec<String>> {
    if code.is_empty() {
        return None;
    }
    let mut sections = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut in_brackets = false;
    let mut chars = code.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                in_quotes = !in_quotes;
                current.push(c);
            }
            '\\' if !in_quotes => {
                current.push(c);
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            '[' if !in_quotes => {
                in_brackets = true;
                current.push(c);
            }
            ']' if !in_quotes => {
                in_brackets = false;
                current.push(c);
            }
            ';' if !in_quotes && !in_brackets => sections.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    if in_quotes || in_brackets {
        return None;
    }
    sections.push(current);
    if sections.len() > 4 {
        return None;
    }
    Some(sections)
}

/// Strip a leading colour modifier and reject every other bracket construct.
/// A condition (`[>100]`), a locale (`[$-409]`) or an elapsed-time token
/// (`[h]`) is outside the bounded grammar.
fn take_color(section: &str) -> Option<(Option<Rgba>, String)> {
    let mut color = None;
    let mut rest = section.trim_start();
    while let Some(stripped) = rest.strip_prefix('[') {
        let end = stripped.find(']')?;
        let body = &stripped[..end];
        color = Some(named_color(body)?);
        rest = &stripped[end + 1..];
    }
    if rest.contains('[') {
        return None;
    }
    Some((color, rest.to_string()))
}

fn named_color(body: &str) -> Option<Rgba> {
    let lower = body.to_ascii_lowercase();
    let named = match lower.as_str() {
        "black" => Some(Rgba(0, 0, 0, 255)),
        "white" => Some(Rgba(255, 255, 255, 255)),
        "red" => Some(Rgba(255, 0, 0, 255)),
        "green" => Some(Rgba(0, 128, 0, 255)),
        "blue" => Some(Rgba(0, 0, 255, 255)),
        "yellow" => Some(Rgba(255, 255, 0, 255)),
        "magenta" => Some(Rgba(255, 0, 255, 255)),
        "cyan" => Some(Rgba(0, 255, 255, 255)),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    let index = lower.strip_prefix("color")?.trim().parse::<usize>().ok()?;
    let entry = INDEXED_COLORS.get(index.checked_sub(1)?)?;
    Rgba::from_argb_hex(entry)
}

fn tokenize(body: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '0' => tokens.push(Token::Zero),
            '#' => tokens.push(Token::Hash),
            '?' => tokens.push(Token::Question),
            '.' => tokens.push(Token::Point),
            ',' => tokens.push(Token::Comma),
            '%' => tokens.push(Token::Percent),
            '@' => tokens.push(Token::Text),
            'E' | 'e' => {
                let explicit_plus = match chars.peek() {
                    Some('+') => true,
                    Some('-') => false,
                    // A bare `E` is a literal, not an exponent marker.
                    _ => {
                        tokens.push(Token::Literal(c));
                        continue;
                    }
                };
                chars.next();
                tokens.push(Token::Exponent { explicit_plus });
            }
            '"' => {
                for quoted in chars.by_ref() {
                    if quoted == '"' {
                        break;
                    }
                    tokens.push(Token::Literal(quoted));
                }
            }
            '\\' => {
                let escaped = chars.next()?;
                tokens.push(Token::Literal(escaped));
            }
            // `_x` reserves the width of `x`. A single space is this
            // formatter's bounded approximation of that reserve.
            '_' => {
                chars.next()?;
                tokens.push(Token::Literal(' '));
            }
            // `*x` repeats `x` to fill the cell. Bounded here as nothing.
            '*' => {
                chars.next()?;
            }
            // Date, time, fraction and elapsed-time grammar: out of scope, and
            // the caller declares the fallback.
            'y' | 'Y' | 'm' | 'M' | 'd' | 'D' | 'h' | 'H' | 's' | 'S' | 'a' | 'A' | 'p' | 'P'
            | '/' => return None,
            '$' | '-' | '+' | '(' | ')' | ':' | ' ' | '\'' => tokens.push(Token::Literal(c)),
            other => tokens.push(Token::Literal(other)),
        }
    }
    Some(tokens)
}

struct Pattern {
    integer_zeros: usize,
    integer_places: usize,
    decimal_tokens: Vec<Token>,
    grouping: bool,
    percent_count: u32,
    exponent: Option<ExponentPattern>,
}

struct ExponentPattern {
    digits: usize,
    explicit_plus: bool,
}

fn analyze(tokens: &[Token]) -> Option<Pattern> {
    let mut integer_zeros = 0usize;
    let mut integer_places = 0usize;
    let mut decimal_tokens = Vec::new();
    let mut exponent_digits = 0usize;
    let mut exponent: Option<ExponentPattern> = None;
    let mut grouping = false;
    let mut percent_count = 0u32;
    let mut seen_point = false;
    let mut seen_digit = false;

    for (index, token) in tokens.iter().enumerate() {
        match token {
            Token::Zero | Token::Hash | Token::Question => {
                seen_digit = true;
                if exponent.is_some() {
                    exponent_digits += 1;
                } else if seen_point {
                    decimal_tokens.push(*token);
                } else {
                    integer_places += 1;
                    if matches!(token, Token::Zero) {
                        integer_zeros += 1;
                    }
                }
            }
            Token::Point => {
                if seen_point || exponent.is_some() {
                    return None;
                }
                seen_point = true;
            }
            Token::Comma => {
                // Only a comma between digit placeholders is grouping. A
                // trailing comma scales by a thousand, which is out of scope.
                let next_is_digit = tokens
                    .get(index + 1)
                    .is_some_and(|t| matches!(t, Token::Zero | Token::Hash | Token::Question));
                if !seen_digit || !next_is_digit {
                    return None;
                }
                grouping = true;
            }
            Token::Percent => percent_count += 1,
            Token::Exponent { explicit_plus } => {
                if exponent.is_some() {
                    return None;
                }
                exponent = Some(ExponentPattern {
                    digits: 0,
                    explicit_plus: *explicit_plus,
                });
            }
            // A numeric section carrying `@` mixes the two grammars.
            Token::Text => return None,
            Token::Literal(_) => {}
        }
    }
    if !seen_digit {
        return None;
    }
    if let Some(exponent) = exponent.as_mut() {
        if exponent_digits == 0 {
            return None;
        }
        exponent.digits = exponent_digits;
    }
    Some(Pattern {
        integer_zeros,
        integer_places,
        decimal_tokens,
        grouping,
        percent_count,
        exponent,
    })
}

fn render_section(section: &str, magnitude: f64, force_negative_sign: bool) -> Option<Formatted> {
    let (color, body) = take_color(section)?;
    let trimmed = body.trim();
    if trimmed.eq_ignore_ascii_case("general") {
        let mut text = general(magnitude);
        if force_negative_sign && magnitude != 0.0 {
            text.insert(0, '-');
        }
        return Some(Formatted { text, color });
    }
    if trimmed.is_empty() {
        // An empty section renders nothing. Excel uses `;;;` to hide values.
        return Some(Formatted {
            text: String::new(),
            color,
        });
    }
    let tokens = tokenize(&body)?;
    // A section may be nothing but literals — `0.0;(0.0);"zero"` is a common
    // shape — in which case no number is placed at all.
    if !tokens
        .iter()
        .any(|t| matches!(t, Token::Zero | Token::Hash | Token::Question))
    {
        let mut out = String::new();
        for token in &tokens {
            match token {
                Token::Literal(c) => out.push(*c),
                Token::Percent => out.push('%'),
                _ => return None,
            }
        }
        return Some(Formatted { text: out, color });
    }
    let pattern = analyze(&tokens)?;

    let mut value = magnitude;
    for _ in 0..pattern.percent_count {
        value *= 100.0;
    }

    let (integer_text, decimal_text, exponent_text) = if let Some(exp) = &pattern.exponent {
        render_scientific(value, &pattern, exp)?
    } else {
        let (integer, decimal) = render_fixed(value, &pattern)?;
        (integer, decimal, None)
    };

    let mut out = String::new();
    if force_negative_sign && (value != 0.0 || !integer_text.is_empty()) {
        out.push('-');
    }
    let mut emitted = false;
    let mut skipping_numeric = false;
    for token in &tokens {
        match token {
            Token::Zero | Token::Hash | Token::Question | Token::Point | Token::Comma => {
                if !emitted {
                    out.push_str(&integer_text);
                    if !decimal_text.is_empty() {
                        out.push('.');
                        out.push_str(&decimal_text);
                    }
                    emitted = true;
                    skipping_numeric = true;
                }
            }
            Token::Exponent { .. } => {
                if let Some(exponent_text) = &exponent_text {
                    out.push_str(exponent_text);
                }
                skipping_numeric = true;
            }
            Token::Percent => {
                out.push('%');
                skipping_numeric = false;
            }
            Token::Literal(c) => {
                let _ = skipping_numeric;
                out.push(*c);
                skipping_numeric = false;
            }
            Token::Text => return None,
        }
    }
    Some(Formatted { text: out, color })
}

/// Round half away from zero at `places` decimals, the way Excel rounds for
/// display. Rust's `{:.n}` rounds half to even on the binary value, which is
/// what made 57.27 come back as 57.0 through umya's path.
fn round_to_places(value: f64, places: usize) -> f64 {
    let factor = 10f64.powi(places as i32);
    (value * factor).round() / factor
}

fn render_fixed(value: f64, pattern: &Pattern) -> Option<(String, String)> {
    let places = pattern.decimal_tokens.len();
    let rounded = round_to_places(value, places);
    if rounded >= MAX_ABS {
        return None;
    }
    let rendered = format!("{rounded:.places$}");
    let (integer_raw, decimal_raw) = match rendered.split_once('.') {
        Some((integer, decimal)) => (integer.to_string(), decimal.to_string()),
        None => (rendered, String::new()),
    };

    let mut integer = integer_raw.trim_start_matches('0').to_string();
    if integer.len() < pattern.integer_zeros {
        let pad = pattern.integer_zeros - integer.len();
        integer.insert_str(0, &"0".repeat(pad));
    }
    // `#,##0.0` with a zero integer part still shows the 0; `#.0` does not.
    if integer.is_empty() && pattern.integer_zeros == 0 && pattern.integer_places > 0 {
        // `#` alone: a zero integer part disappears only when there are
        // decimals to carry the value.
        if places == 0 {
            integer.push('0');
        }
    }
    if pattern.grouping {
        integer = group_thousands(&integer);
    }

    // Trailing `#` placeholders drop trailing zeros; `?` pads with a space.
    let mut decimal: Vec<char> = decimal_raw.chars().collect();
    for (index, token) in pattern.decimal_tokens.iter().enumerate().rev() {
        match token {
            Token::Hash if decimal.get(index) == Some(&'0') => {
                decimal.truncate(index);
            }
            Token::Question if decimal.get(index) == Some(&'0') => {
                if index + 1 == decimal.len() {
                    decimal[index] = ' ';
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    Some((integer, decimal.into_iter().collect()))
}

fn render_scientific(
    value: f64,
    pattern: &Pattern,
    exp: &ExponentPattern,
) -> Option<(String, String, Option<String>)> {
    let places = pattern.decimal_tokens.len();
    let mantissa_digits = pattern.integer_places.max(1);
    let mut exponent = if value == 0.0 {
        0i32
    } else {
        value.abs().log10().floor() as i32 - (mantissa_digits as i32 - 1)
    };
    let mut mantissa = if value == 0.0 {
        0.0
    } else {
        value / 10f64.powi(exponent)
    };
    // Rounding the mantissa can carry it past its digit count (9.99 -> 10.0).
    let limit = 10f64.powi(mantissa_digits as i32);
    if round_to_places(mantissa.abs(), places) >= limit {
        mantissa /= 10.0;
        exponent += 1;
    }
    let rendered = format!("{:.*}", places, round_to_places(mantissa, places));
    let (integer, decimal) = match rendered.split_once('.') {
        Some((integer, decimal)) => (integer.to_string(), decimal.to_string()),
        None => (rendered, String::new()),
    };
    let sign = if exponent < 0 {
        '-'
    } else if exp.explicit_plus {
        '+'
    } else {
        // `E-00` prints nothing for a positive exponent.
        '\0'
    };
    let mut exponent_text = String::from("E");
    if sign != '\0' {
        exponent_text.push(sign);
    }
    exponent_text.push_str(&format!("{:0width$}", exponent.abs(), width = exp.digits));
    Some((integer, decimal, Some(exponent_text)))
}

fn group_thousands(digits: &str) -> String {
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let count = digits.len();
    for (index, c) in digits.chars().enumerate() {
        if index > 0 && (count - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Excel's `General`: up to 11 significant digits, no trailing zeros, and
/// scientific notation outside that range.
pub fn general(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    if !(-5..11).contains(&magnitude) {
        // Rust writes a negative exponent with its sign and a positive one
        // without; Excel always writes the sign.
        let rendered = format!("{value:.5E}");
        return match rendered.split_once('E') {
            Some((mantissa, exponent)) if !exponent.starts_with('-') => {
                format!("{mantissa}E+{exponent}")
            }
            _ => rendered,
        };
    }
    let places = (10 - magnitude).clamp(0, 10) as usize;
    let rendered = format!("{value:.places$}");
    if rendered.contains('.') {
        rendered
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else {
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: f64, code: &str) -> String {
        format_number(value, code).unwrap().text
    }

    #[test]
    fn multi_section_percent_is_the_bake_off_regression() {
        // umya returned "57.0%" here; Excel, LibreOffice, BetterOffice and
        // readany-render all give "57.3%".
        let formatted = format_number(0.5727, "0.0%;[Red]-0.0%").unwrap();
        assert_eq!(formatted.text, "57.3%");
        assert_eq!(formatted.color, None);

        // umya returned "3.0%" here, dropping both the sign and the colour.
        let negative = format_number(-0.032, "0.0%;[Red]-0.0%").unwrap();
        assert_eq!(negative.text, "-3.2%");
        assert_eq!(negative.color, Some(Rgba(255, 0, 0, 255)));
    }

    #[test]
    fn single_section_supplies_its_own_negative_sign() {
        assert_eq!(text(-1234.5, "#,##0.00"), "-1,234.50");
        assert_eq!(text(1234.5, "#,##0.00"), "1,234.50");
    }

    #[test]
    fn three_sections_select_by_sign() {
        let code = "0.0;(0.0);\"zero\"";
        assert_eq!(text(2.25, code), "2.3");
        assert_eq!(text(-2.25, code), "(2.3)");
        assert_eq!(text(0.0, code), "zero");
    }

    #[test]
    fn grouping_and_zero_padding() {
        assert_eq!(text(1234567.0, "#,##0"), "1,234,567");
        assert_eq!(text(7.0, "000"), "007");
        assert_eq!(text(0.0, "#,##0"), "0");
    }

    #[test]
    fn hash_placeholders_drop_trailing_zeros() {
        assert_eq!(text(1.5, "0.##"), "1.5");
        assert_eq!(text(1.0, "0.##"), "1");
        assert_eq!(text(1.25, "0.##"), "1.25");
        assert_eq!(text(1.5, "0.00"), "1.50");
    }

    #[test]
    fn rounds_half_away_from_zero_like_excel() {
        // Rust's `{:.2}` rounds this half-to-even and gives 0.12.
        assert_eq!(text(0.125, "0.00"), "0.13");
        assert_eq!(text(2.5, "0"), "3");
    }

    #[test]
    fn quoted_literals_and_currency() {
        assert_eq!(text(1234.0, "\"$\"#,##0"), "$1,234");
        assert_eq!(text(1234.0, "$#,##0"), "$1,234");
        assert_eq!(text(5.0, "0\" units\""), "5 units");
        assert_eq!(text(5.0, "\\#0"), "#5");
    }

    #[test]
    fn scientific_notation() {
        assert_eq!(text(12345.6789, "0.00E+00"), "1.23E+04");
        assert_eq!(text(0.000123, "0.00E+00"), "1.23E-04");
        assert_eq!(text(12345.6789, "0.00E-00"), "1.23E04");
    }

    #[test]
    fn general_matches_excel_significance() {
        assert_eq!(general(0.5727), "0.5727");
        assert_eq!(general(1.0), "1");
        assert_eq!(general(-42.0), "-42");
        assert_eq!(text(0.5727, "General"), "0.5727");
    }

    #[test]
    fn hidden_sections_render_nothing() {
        assert_eq!(text(42.0, ";;;"), "");
    }

    #[test]
    fn color_n_indexes_the_legacy_palette() {
        let formatted = format_number(-1.0, "0;[Color 3]0").unwrap();
        assert_eq!(formatted.color, Some(Rgba(0xff, 0x00, 0x00, 255)));
    }

    #[test]
    fn unsupported_grammar_falls_back() {
        // Dates, conditions, locales, elapsed time, fractions and comma
        // scaling all decline rather than guess.
        assert!(format_number(45000.0, "yyyy-mm-dd").is_none());
        assert!(format_number(1.0, "[>100]0.0;0.0").is_none());
        assert!(format_number(1.0, "[$-409]0.0").is_none());
        assert!(format_number(1.0, "[h]:mm:ss").is_none());
        assert!(format_number(1.0, "# ?/?").is_none());
        assert!(format_number(1_000_000.0, "0.0,,").is_none());
        assert!(format_number(f64::NAN, "0.0").is_none());
        assert!(format_number(1e16, "0.0").is_none());
        assert!(format_number(1.0, "").is_none());
    }

    #[test]
    fn text_section_applies_to_strings() {
        let formatted = format_text("hi", "0.0;;;\"<\"@\">\"").unwrap();
        assert_eq!(formatted.text, "<hi>");
        let untouched = format_text("hi", "0.0").unwrap();
        assert_eq!(untouched.text, "hi");
    }
}
