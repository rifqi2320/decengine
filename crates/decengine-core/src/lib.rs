//! Stable domain types shared by every decengine frontend and backend.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const MAX_QUESTIONS: usize = 64;
pub const MAX_CANDIDATES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    InvalidQuestion,
    ModelNotFound,
    UnsupportedModel,
    ModelLoadFailed,
    ContextTooLong,
    EngineError,
    OutOfMemory,
    InvalidManifest,
    InternalError,
}

#[derive(Debug, Error)]
#[error("{code:?}: {message}")]
pub struct DecengineError {
    pub code: ErrorCode,
    pub message: String,
}

impl DecengineError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidRequest, message)
    }

    pub fn invalid_question(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidQuestion, message)
    }
}

pub type Result<T> = std::result::Result<T, DecengineError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

impl DecisionRequest {
    /// Validates request-wide limits and every contained question.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the state is null, the question collection is empty or too
    /// large, or any question is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.state.is_null() {
            return Err(DecengineError::invalid_request("state must not be null"));
        }
        if self.questions.is_empty() {
            return Err(DecengineError::invalid_request(
                "at least one question is required",
            ));
        }
        if self.questions.len() > MAX_QUESTIONS {
            return Err(DecengineError::invalid_request(format!(
                "at most {MAX_QUESTIONS} questions are allowed"
            )));
        }

        for (name, question) in &self.questions {
            if name.trim().is_empty() {
                return Err(DecengineError::invalid_question(
                    "question names must not be blank",
                ));
            }
            question.validate(name)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    Choice {
        prompt: String,
        options: BTreeMap<String, String>,
    },
    Noul {
        prompt: String,
    },
    Score {
        prompt: String,
        levels: Vec<ScoreLevel>,
    },
}

impl Question {
    /// Validates the prompt and the constraints specific to this question variant.
    ///
    /// # Errors
    ///
    /// Returns [`DecengineError`] when the prompt, candidates, or score levels are invalid.
    pub fn validate(&self, name: &str) -> Result<()> {
        match self {
            Self::Choice { prompt, options } => {
                validate_prompt(name, prompt)?;
                if !(2..=MAX_CANDIDATES).contains(&options.len()) {
                    return Err(DecengineError::invalid_question(format!(
                        "choice {name:?} requires 2..={MAX_CANDIDATES} options"
                    )));
                }
                for (label, criterion) in options {
                    if label.trim().is_empty() || criterion.trim().is_empty() {
                        return Err(DecengineError::invalid_question(format!(
                            "choice {name:?} has a blank option label or criterion"
                        )));
                    }
                }
            }
            Self::Noul { prompt } => validate_prompt(name, prompt)?,
            Self::Score { prompt, levels } => {
                validate_prompt(name, prompt)?;
                if !(2..=MAX_CANDIDATES).contains(&levels.len()) {
                    return Err(DecengineError::invalid_question(format!(
                        "score {name:?} requires 2..={MAX_CANDIDATES} levels"
                    )));
                }
                for level in levels {
                    if level.label.trim().is_empty() || level.criterion.trim().is_empty() {
                        return Err(DecengineError::invalid_question(format!(
                            "score {name:?} has a blank level label or criterion"
                        )));
                    }
                    if !level.value.is_finite() {
                        return Err(DecengineError::invalid_question(format!(
                            "score {name:?} has a non-finite level value"
                        )));
                    }
                }
                if levels.windows(2).any(|pair| pair[0].value >= pair[1].value) {
                    return Err(DecengineError::invalid_question(format!(
                        "score {name:?} level values must be strictly increasing"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn validate_prompt(name: &str, prompt: &str) -> Result<()> {
    if prompt.trim().is_empty() {
        return Err(DecengineError::invalid_question(format!(
            "question {name:?} has a blank prompt"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreLevel {
    pub label: String,
    pub criterion: String,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub results: BTreeMap<String, DecisionResult>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionResult {
    Choice {
        selected: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Noul {
        value: f32,
        confidence: f32,
    },
    Score {
        value: f32,
        distribution: BTreeMap<String, f32>,
        legend: BTreeMap<String, String>,
        confidence: f32,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_characters: usize,
    pub questions: usize,
    pub candidates: usize,
    pub candidate_cache_hits: usize,
    pub candidate_cache_misses: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: ErrorCode,
    pub message: String,
}

impl From<&DecengineError> for ErrorEnvelope {
    fn from(value: &DecengineError) -> Self {
        Self {
            error: ErrorBody {
                code: value.code,
                message: value.message.clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(question: Question) -> DecisionRequest {
        DecisionRequest {
            model: None,
            state: serde_json::json!({"ticket": "refund"}),
            questions: BTreeMap::from([("route".to_owned(), question)]),
        }
    }

    #[test]
    fn validates_choice() {
        let valid = request(Question::Choice {
            prompt: "Where?".to_owned(),
            options: BTreeMap::from([
                ("billing".to_owned(), "refunds".to_owned()),
                ("sales".to_owned(), "pricing".to_owned()),
            ]),
        });
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn rejects_single_choice_option() {
        let invalid = request(Question::Choice {
            prompt: "Where?".to_owned(),
            options: BTreeMap::from([("billing".to_owned(), "refunds".to_owned())]),
        });
        assert_eq!(
            invalid.validate().unwrap_err().code,
            ErrorCode::InvalidQuestion
        );
    }

    #[test]
    fn rejects_unordered_score_values() {
        let invalid = request(Question::Score {
            prompt: "Severity?".to_owned(),
            levels: vec![
                ScoreLevel {
                    label: "high".to_owned(),
                    criterion: "critical".to_owned(),
                    value: 2.0,
                },
                ScoreLevel {
                    label: "low".to_owned(),
                    criterion: "minor".to_owned(),
                    value: 1.0,
                },
            ],
        });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn error_envelope_is_stable() {
        let error = DecengineError::invalid_request("bad input");
        let json = serde_json::to_value(ErrorEnvelope::from(&error)).unwrap();
        assert_eq!(json["error"]["code"], "invalid_request");
        assert_eq!(json["error"]["message"], "bad input");
    }
}
