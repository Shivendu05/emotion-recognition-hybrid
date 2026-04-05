import numpy as np
import skfuzzy as fuzz

def fuzzy_emotion(score):
    x = np.arange(0, 1.1, 0.1)

    low = fuzz.trimf(x, [0, 0, 0.5])
    medium = fuzz.trimf(x, [0, 0.5, 1])
    high = fuzz.trimf(x, [0.5, 1, 1])

    low_val = fuzz.interp_membership(x, low, score)
    med_val = fuzz.interp_membership(x, medium, score)
    high_val = fuzz.interp_membership(x, high, score)

    if high_val > med_val and high_val > low_val:
        return "Strong Emotion"
    elif med_val > low_val:
        return "Moderate Emotion"
    else:
        return "Weak Emotion"