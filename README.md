# Two-visit CRDM
**Code by:**

Santiago Guardo & Ricardo Pizarro

**Supervision:**

Ricardo Pizarro and Silvia Lopez
## This script creates a new choice set given ADO o participant's input.

- It uses the parameters from the utility model and creates a new choice set that symmetrically samples the **subjective value difference**. 

 $SVlott = (p - \beta \cdot \frac{A}{2}) \cdot V^\alpha and SVsafe = V^\alpha$

- This will help increase CASANDRE model fit by having enogh and equal amount observations in each side of the point of subjective equality. 

- For each probability level:
    - We want to make 3 trials around the point of subjective equality.
    - We want 2 extreme trials.
        - Winning $50 for each probability level
        - Loosing $50 for each probability level
    - We want 4 intermediate trials.

# 1. How to use it
- If you are making a calibration visit and then a tailored choiceset visit
    - Run the utility model: https://github.com/CDN-Lab/IDM_model 
    - Modify the input of this cript so that it reads the parameters of the model
    - Run the script
- For ADO in person data, it's all going to be inbedded in psychopy. 




