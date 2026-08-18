# User Study Protocol

## TrustLens: Evaluating Explanation Methods for Zero Trust Authentication

### Study Overview
- **Purpose**: Evaluate how different XAI explanation methods affect security analyst decision-making
- **IRB Status**: [TODO: Submit for IRB approval]
- **Target Participants**: 20-30 security analysts with 1+ years experience

### Study Design
- **Type**: Within-subjects, counterbalanced
- **Independent Variable**: Explanation type (SHAP, LIME, Anchor, None)
- **Dependent Variables**:
  - Decision accuracy (correct identification of threat level)
  - Decision time (seconds to reach decision)
  - Subjective trust rating (1-7 Likert scale)
  - Explanation usefulness rating (1-7 Likert scale)

### Procedure
1. **Introduction** (5 min): Brief on ZTA and study goals
2. **Training** (10 min): Familiarize with dashboard and explanation types
3. **Task Phase** (30 min): Evaluate 20 authentication decisions (5 per condition)
4. **Questionnaire** (10 min): Post-task survey
5. **Debrief** (5 min): Explain study goals and collect feedback

### Scenarios
Each participant evaluates 20 authentication events:
- 5 with SHAP explanations
- 5 with LIME explanations
- 5 with Anchor explanations
- 5 with no explanation (baseline)

### Recruitment
- Posted on university channels and professional security forums
- Eligibility: 1+ year cybersecurity experience
- Compensation: [TODO: Determine compensation]

### Data Collection
- Responses saved to `results/raw_responses.csv`
- All data anonymized (no PII collected)
