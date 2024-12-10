# EMAP Baseline Results Repository  

Welcome to the **EMAP Baseline Results Repository**! This repository contains the code for generating baseline results for our **EMAP (Emotional Arousal Pattern) dataset challenge**, focusing on both classification and regression tasks related to predicting emotional arousal ratings from the EMAP dataset.

---

## Dataset & Objective  

This repository provides scripts and methods used to compute baseline results for the EMAP dataset challenge. The EMAP dataset is designed to model subjective and psychophysiological responses to affective stimuli. This repository serves as a reference point for arousal classification and regression tasks.

---

## Citation  

If you use the EMAP dataset or methods from this repository for predicting arousal ratings (classification or regression), please reference the following paper:  

1. **Eisenbarth, H., Oxner, M., Shehu, H. A., Gastrell, T., Walsh, A., Browne, W. N., & Xue, B. (2024).** Emotional arousal pattern (EMAP): A new database for modeling momentary subjective and psychophysiological responding to affective stimuli. *Psychophysiology, 61(2), e14446.*  

For related work focusing on predicting heart rate and skin conductance responses:  

2. **Shehu, H. A., Oxner, M., Browne, W. N., & Eisenbarth, H. (2023).** Prediction of moment‐by‐moment heart rate and skin conductance changes in the context of varying emotional arousal. *Psychophysiology, 60(9), e14303.*  

---

## Contents (Each code performs)

- **Dataset Processing:** Preprocess and prepare the EMAP dataset for model evaluation.  
- **Baseline Results Generation:** Generate baseline results for classification or regression predictions.  
- **Model References:** Implementation for baseline comparison tasks related to arousal classification and regression.  
- **Feature Selection and Model Saving:** Save models before and after feature selection and save the results of selected features along with their threshold values.

---

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/emapchallenge/EMAP-baseline.git
   ```
2. Navigate to the directory:  
   ```bash
   cd EMAP-baseline-main
   ```
3. Install requited dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage  

To run the baseline arousal regression script, execute the following:  

```bash
python arousal_regression.py
```

## Acknowledgments  

This work utilized the EMAP dataset, as referenced in the papers provided above. We appreciate the contributions of all researchers involved in the development and exploration of this dataset.  

For questions, feedback, or contributions, please contact: [harisu.shehu@ecs.vuw.ac.nz](mailto:harisu.shehu@ecs.vuw.ac.nz).  

**The EMAP Research Team**
