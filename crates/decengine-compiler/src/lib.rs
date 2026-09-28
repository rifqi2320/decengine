//! Versioned compilation from public decision questions into an engine batch.

use std::collections::BTreeMap;

use decengine_core::{DecisionRequest, Question, Result};
use decengine_engine::{DecisionKind, EngineCandidate, EngineDecision, EngineDecisionBatch};
use decengine_models::ModelProfile;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const COMPILER_VERSION: &str = "1";
pub const TEMPLATE_VERSION: &str = "1";

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledBatch {
    pub engine_batch: EngineDecisionBatch,
    pub input_characters: usize,
    pub candidates: usize,
}

/// Validates and compiles a public request into the backend-neutral engine representation.
///
/// # Errors
///
/// Returns [`decengine_core::DecengineError`] when the request fails domain validation.
pub fn compile(request: &DecisionRequest, profile: &ModelProfile) -> Result<CompiledBatch> {
    request.validate()?;
    let state = canonical_state(&request.state);
    let mut decisions = Vec::with_capacity(request.questions.len());
    let mut candidate_count = 0;
    let mut input_characters = state.len();

    for (key, question) in &request.questions {
        let (kind, prompt, candidates) = compile_question(question, profile);
        candidate_count += candidates.len();
        input_characters += prompt.len();
        input_characters += candidates.iter().map(|item| item.text.len()).sum::<usize>();

        let raw_query = format!("Decision: {prompt}\nState: {state}");
        let instruction = apply_instruction_template(profile, prompt, question_kind(question));
        let query = apply_query_template(profile, &instruction, &raw_query);
        let cache_key = cache_key(profile, &kind, prompt, &candidates);
        decisions.push(EngineDecision {
            key: key.clone(),
            kind,
            query,
            candidates,
            cache_key,
        });
    }

    Ok(CompiledBatch {
        engine_batch: EngineDecisionBatch { decisions },
        input_characters,
        candidates: candidate_count,
    })
}

fn compile_question<'a>(
    question: &'a Question,
    profile: &ModelProfile,
) -> (DecisionKind, &'a str, Vec<EngineCandidate>) {
    match question {
        Question::Choice { prompt, options } => (
            DecisionKind::Choice,
            prompt,
            options
                .iter()
                .map(|(label, criterion)| EngineCandidate {
                    id: label.clone(),
                    text: apply_candidate_template(
                        profile,
                        &profile
                            .choice_candidate_template
                            .replace("{label}", label)
                            .replace("{criterion}", criterion),
                    ),
                    score_value: None,
                })
                .collect(),
        ),
        Question::Noul { prompt } => (
            DecisionKind::Noul,
            prompt,
            vec![
                EngineCandidate {
                    id: "unsupported".to_owned(),
                    text: apply_candidate_template(profile, &profile.noul_unsupported),
                    score_value: Some(0.0),
                },
                EngineCandidate {
                    id: "supported".to_owned(),
                    text: apply_candidate_template(profile, &profile.noul_supported),
                    score_value: Some(1.0),
                },
            ],
        ),
        Question::Score { prompt, levels } => (
            DecisionKind::Score,
            prompt,
            levels
                .iter()
                .map(|level| EngineCandidate {
                    id: level.label.clone(),
                    text: apply_candidate_template(
                        profile,
                        &profile
                            .score_candidate_template
                            .replace("{label}", &level.label)
                            .replace("{criterion}", &level.criterion),
                    ),
                    score_value: Some(level.value),
                })
                .collect(),
        ),
    }
}

fn apply_query_template(profile: &ModelProfile, instruction: &str, text: &str) -> String {
    profile
        .query_template
        .replace("{instruction}", instruction)
        .replace("{text}", text)
}

fn apply_instruction_template(profile: &ModelProfile, prompt: &str, kind: &str) -> String {
    profile
        .query_instruction_template
        .replace("{prompt}", prompt)
        .replace("{kind}", kind)
}

fn question_kind(question: &Question) -> &'static str {
    match question {
        Question::Choice { .. } => {
            "choose the single best-matching option from the reported issue and candidate scopes; do not infer facts absent from the case"
        }
        Question::Noul { .. } => {
            "decide whether same-day action is needed to prevent likely harm, material interruption, or a near-term missed deadline; an available safe workaround or no concrete near-term consequence favors no; emphatic wording alone is not evidence"
        }
        Question::Score { .. } => {
            "select the highest impact supported if unresolved; distinguish routine/no concrete near-term risk, localized disruption or safe workaround, and credible immediate or broad serious risk"
        }
    }
}

fn apply_candidate_template(profile: &ModelProfile, text: &str) -> String {
    profile.candidate_template.replace("{text}", text)
}

fn cache_key(
    profile: &ModelProfile,
    kind: &DecisionKind,
    prompt: &str,
    candidates: &[EngineCandidate],
) -> String {
    let mut hasher = Sha256::new();
    for part in [
        profile.id.as_str(),
        profile.profile_version.as_str(),
        COMPILER_VERSION,
        TEMPLATE_VERSION,
        profile.query_template.as_str(),
        profile.candidate_template.as_str(),
        profile.query_instruction_template.as_str(),
        profile.choice_candidate_template.as_str(),
        profile.score_candidate_template.as_str(),
        prompt,
    ] {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hasher.update(format!("{kind:?}").as_bytes());
    for candidate in candidates {
        hasher.update((candidate.id.len() as u64).to_le_bytes());
        hasher.update(candidate.id.as_bytes());
        hasher.update((candidate.text.len() as u64).to_le_bytes());
        hasher.update(candidate.text.as_bytes());
        if let Some(value) = candidate.score_value {
            hasher.update(value.to_le_bytes());
        }
    }
    hex::encode(hasher.finalize())
}

fn canonical_state(value: &Value) -> String {
    serde_json::to_string(&sort_value(value)).expect("serializing serde_json::Value cannot fail")
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), sort_value(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(sort_value).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decengine_core::ScoreLevel;
    use decengine_models::ModelRegistry;

    fn profile() -> ModelProfile {
        ModelRegistry::bundled()
            .unwrap()
            .get("Qwen/Qwen3-Embedding-0.6B")
            .unwrap()
            .clone()
    }

    #[test]
    fn compiles_all_question_kinds() {
        let request = DecisionRequest {
            model: None,
            state: serde_json::json!({"body": "refund today"}),
            questions: BTreeMap::from([
                (
                    "route".to_owned(),
                    Question::Choice {
                        prompt: "Which team?".to_owned(),
                        options: BTreeMap::from([
                            ("billing".to_owned(), "refunds".to_owned()),
                            ("sales".to_owned(), "pricing".to_owned()),
                        ]),
                    },
                ),
                (
                    "severity".to_owned(),
                    Question::Score {
                        prompt: "How severe?".to_owned(),
                        levels: vec![
                            ScoreLevel {
                                label: "low".to_owned(),
                                criterion: "minor".to_owned(),
                                value: 0.0,
                            },
                            ScoreLevel {
                                label: "high".to_owned(),
                                criterion: "critical".to_owned(),
                                value: 1.0,
                            },
                        ],
                    },
                ),
                (
                    "urgent".to_owned(),
                    Question::Noul {
                        prompt: "Does this require attention today?".to_owned(),
                    },
                ),
            ]),
        };
        let compiled = compile(&request, &profile()).unwrap();
        assert_eq!(compiled.engine_batch.decisions.len(), 3);
        assert_eq!(compiled.candidates, 6);
        assert!(
            compiled
                .engine_batch
                .decisions
                .iter()
                .all(|decision| decision.cache_key.len() == 64)
        );
    }

    #[test]
    fn noul_rubric_text_is_profile_parameterized() {
        let request = DecisionRequest {
            model: None,
            state: serde_json::json!({"deadline": "today"}),
            questions: BTreeMap::from([(
                "urgent".to_owned(),
                Question::Noul {
                    prompt: "Urgent?".to_owned(),
                },
            )]),
        };
        let mut profile = profile();
        profile.noul_unsupported = "No same-day action".to_owned();
        profile.noul_supported = "Same-day action".to_owned();
        let compiled = compile(&request, &profile).unwrap();
        let candidates = &compiled.engine_batch.decisions[0].candidates;
        assert_eq!(candidates[0].text, "No same-day action");
        assert_eq!(candidates[1].text, "Same-day action");
    }

    #[test]
    fn task_instruction_and_candidate_phrasing_are_opt_in() {
        let request = DecisionRequest {
            model: None,
            state: serde_json::json!({"issue": "a thing"}),
            questions: BTreeMap::from([(
                "route".to_owned(),
                Question::Choice {
                    prompt: "Where?".to_owned(),
                    options: BTreeMap::from([
                        ("a".to_owned(), "criterion a".to_owned()),
                        ("b".to_owned(), "criterion b".to_owned()),
                    ]),
                },
            )]),
        };
        let mut profile = profile();
        profile.query_instruction_template = "Task for {kind}: {prompt}".to_owned();
        profile.choice_candidate_template = "Function {label}. Scope: {criterion}".to_owned();
        let compiled = compile(&request, &profile).unwrap();
        let decision = &compiled.engine_batch.decisions[0];
        assert!(decision.query.starts_with("Instruct: Task for choose the single best-matching option from the reported issue and candidate scopes; do not infer facts absent from the case: Where?\n"));
        assert_eq!(
            decision.candidates[0].text,
            "Function a. Scope: criterion a"
        );
    }

    #[test]
    fn cache_key_ignores_state_but_changes_with_criteria() {
        let build = |state: &str, criterion: &str| DecisionRequest {
            model: None,
            state: serde_json::json!(state),
            questions: BTreeMap::from([(
                "route".to_owned(),
                Question::Choice {
                    prompt: "Which?".to_owned(),
                    options: BTreeMap::from([
                        ("a".to_owned(), criterion.to_owned()),
                        ("b".to_owned(), "other".to_owned()),
                    ]),
                },
            )]),
        };
        let first = compile(&build("one", "alpha"), &profile()).unwrap();
        let second = compile(&build("two", "alpha"), &profile()).unwrap();
        let changed = compile(&build("one", "changed"), &profile()).unwrap();
        assert_eq!(
            first.engine_batch.decisions[0].cache_key,
            second.engine_batch.decisions[0].cache_key
        );
        assert_ne!(
            first.engine_batch.decisions[0].cache_key,
            changed.engine_batch.decisions[0].cache_key
        );
    }

    #[test]
    fn canonicalizes_object_key_order() {
        let left = serde_json::json!({"b": 2, "a": {"z": 1, "x": 0}});
        let right = serde_json::json!({"a": {"x": 0, "z": 1}, "b": 2});
        assert_eq!(canonical_state(&left), canonical_state(&right));
    }
}
