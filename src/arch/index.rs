use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

/// One named, zero-based index range.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexDomain {
    pub name: String,
    pub size: u64,
}

impl IndexDomain {
    pub fn new(name: impl Into<String>, size: u64) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

/// Integer affine expression used by memory endpoints.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AffineExpression {
    Constant(i64),
    Variable(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(i64, Box<Self>),
    FloorDiv(Box<Self>, i64),
    CeilDiv(Box<Self>, i64),
    Mod(Box<Self>, i64),
}

impl AffineExpression {
    pub fn parse(input: &str) -> Result<Self, EndpointParseError> {
        ExpressionParser::new(input).parse()
    }

    pub fn evaluate(&self, values: &BTreeMap<String, i64>) -> Option<i64> {
        match self {
            Self::Constant(value) => Some(*value),
            Self::Variable(name) => values.get(name).copied(),
            Self::Add(lhs, rhs) => Some(lhs.evaluate(values)? + rhs.evaluate(values)?),
            Self::Sub(lhs, rhs) => Some(lhs.evaluate(values)? - rhs.evaluate(values)?),
            Self::Mul(factor, expr) => Some(factor * expr.evaluate(values)?),
            Self::FloorDiv(expr, divisor) => {
                let value = expr.evaluate(values)?;
                if *divisor <= 0 {
                    None
                } else {
                    Some(value.div_euclid(*divisor))
                }
            }
            Self::CeilDiv(expr, divisor) => {
                let value = expr.evaluate(values)?;
                if *divisor <= 0 {
                    None
                } else {
                    Some(-(-value).div_euclid(*divisor))
                }
            }
            Self::Mod(expr, modulus) => {
                if *modulus <= 0 {
                    None
                } else {
                    Some(expr.evaluate(values)?.rem_euclid(*modulus))
                }
            }
        }
    }

    pub fn variables(&self) -> BTreeSet<String> {
        let mut variables = BTreeSet::new();
        self.collect_variables(&mut variables);
        variables
    }

    fn collect_variables(&self, variables: &mut BTreeSet<String>) {
        match self {
            Self::Variable(name) => {
                variables.insert(name.clone());
            }
            Self::Add(lhs, rhs) | Self::Sub(lhs, rhs) => {
                lhs.collect_variables(variables);
                rhs.collect_variables(variables);
            }
            Self::Mul(_, expr)
            | Self::FloorDiv(expr, _)
            | Self::CeilDiv(expr, _)
            | Self::Mod(expr, _) => expr.collect_variables(variables),
            Self::Constant(_) => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointParseError {
    pub message: String,
    pub position: usize,
}

impl EndpointParseError {
    fn new(input: &str, rest: &str, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position: input.len().saturating_sub(rest.len()),
        }
    }
}

impl fmt::Display for EndpointParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "endpoint expression parse error at {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for EndpointParseError {}

struct ExpressionParser<'a> {
    input: &'a str,
    rest: &'a str,
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, rest: input }
    }

    fn parse(mut self) -> Result<AffineExpression, EndpointParseError> {
        let expression = self.parse_add_sub()?;
        self.skip_space();
        if self.rest.is_empty() {
            Ok(expression)
        } else {
            Err(self.error("unexpected trailing input"))
        }
    }

    fn parse_add_sub(&mut self) -> Result<AffineExpression, EndpointParseError> {
        let mut lhs = self.parse_mul_div()?;
        loop {
            self.skip_space();
            if self.consume("+") {
                lhs = AffineExpression::Add(Box::new(lhs), Box::new(self.parse_mul_div()?));
            } else if self.consume("-") {
                lhs = AffineExpression::Sub(Box::new(lhs), Box::new(self.parse_mul_div()?));
            } else {
                return Ok(lhs);
            }
        }
    }

    fn parse_mul_div(&mut self) -> Result<AffineExpression, EndpointParseError> {
        let mut lhs = self.parse_atom()?;
        loop {
            self.skip_space();
            if self.consume_keyword("floordiv") {
                let divisor = self.parse_positive_constant("floordiv")?;
                lhs = AffineExpression::FloorDiv(Box::new(lhs), divisor);
            } else if self.consume_keyword("ceildiv") {
                let divisor = self.parse_positive_constant("ceildiv")?;
                lhs = AffineExpression::CeilDiv(Box::new(lhs), divisor);
            } else if self.consume_keyword("mod") || self.consume("%") {
                let modulus = self.parse_positive_constant("mod")?;
                lhs = AffineExpression::Mod(Box::new(lhs), modulus);
            } else if self.consume("*") {
                let rhs = self.parse_atom()?;
                lhs = match (lhs, rhs) {
                    (AffineExpression::Constant(c), expression)
                    | (expression, AffineExpression::Constant(c)) => {
                        AffineExpression::Mul(c, Box::new(expression))
                    }
                    _ => return Err(self.error("affine multiplication requires one constant")),
                };
            } else {
                return Ok(lhs);
            }
        }
    }

    fn parse_atom(&mut self) -> Result<AffineExpression, EndpointParseError> {
        self.skip_space();
        if self.consume("(") {
            let expression = self.parse_add_sub()?;
            self.skip_space();
            if !self.consume(")") {
                return Err(self.error("expected ')'"));
            }
            return Ok(expression);
        }
        if self.consume("-") {
            return Ok(AffineExpression::Mul(-1, Box::new(self.parse_atom()?)));
        }
        if let Some(number) = self.take_while(|c| c.is_ascii_digit()) {
            return number
                .parse::<i64>()
                .map(AffineExpression::Constant)
                .map_err(|_| self.error("integer is out of range"));
        }
        if let Some(identifier) = self.take_while(|c| c.is_ascii_alphanumeric() || c == '_') {
            if identifier
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
            {
                return Err(self.error("identifier cannot start with a digit"));
            }
            return Ok(AffineExpression::Variable(identifier.to_string()));
        }
        Err(self.error("expected an integer, variable, or parenthesized expression"))
    }

    fn parse_positive_constant(&mut self, operator: &str) -> Result<i64, EndpointParseError> {
        self.skip_space();
        let Some(number) = self.take_while(|c| c.is_ascii_digit()) else {
            return Err(self.error(format!("{operator} requires a positive constant divisor")));
        };
        let value = number
            .parse::<i64>()
            .map_err(|_| self.error("integer is out of range"))?;
        if value <= 0 {
            return Err(self.error(format!("{operator} divisor must be positive")));
        }
        Ok(value)
    }

    fn skip_space(&mut self) {
        self.rest = self.rest.trim_start();
    }

    fn consume(&mut self, token: &str) -> bool {
        if let Some(rest) = self.rest.strip_prefix(token) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        let Some(rest) = self.rest.strip_prefix(keyword) else {
            return false;
        };
        if rest
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return false;
        }
        self.rest = rest;
        true
    }

    fn take_while(&mut self, predicate: impl Fn(char) -> bool) -> Option<&'a str> {
        let length = self
            .rest
            .char_indices()
            .take_while(|(_, c)| predicate(*c))
            .last()
            .map_or(0, |(offset, c)| offset + c.len_utf8());
        if length == 0 {
            None
        } else {
            let (value, rest) = self.rest.split_at(length);
            self.rest = rest;
            Some(value)
        }
    }

    fn error(&self, message: impl Into<String>) -> EndpointParseError {
        EndpointParseError::new(self.input, self.rest, message)
    }
}
