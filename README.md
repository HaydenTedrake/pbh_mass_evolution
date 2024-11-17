# Primordial Black Hole Mass Evolution Research

### Faculty Advisor  
**Peter Fisher**

### Direct Supervisor  
**Alexandra Klipfel**

---

## Research Overview

This project focuses on the mass evolution of **Primordial Black Holes (PBHs)** using Python. It determines the mass of a PBH at the location of its explosion based on its formation mass. The research uses **Carr's differential equation**, a key model for understanding PBH mass evolution over time in the absence of accretion:

\[
\frac{dm}{dt} \propto \frac{f(m)}{m^2}
\]

This equation describes PBH mass loss as a function of time, where \( f(m) \) is the number of particle species that can be produced by a PBH of mass \( m \). An approximation for \( f(m) \), developed by my mentor Alexandra Klipfel, is currently used in this study.

---

## Methodology

1. **Simplifying the PBH Path**  
   The PBH's trajectory is approximated as straight-line motion in the x-direction relative to the Sun. Using kinematics and assuming a velocity of approximately 220 km/s, the travel time of the PBH between its formation location and explosion site is calculated.

2. **Integration and Validation**  
   - The calculated travel time is used to integrate backward through Carr’s differential equation, obtaining the PBH's mass at the explosion location.
   - The results are validated using Carr’s equation.

3. **Future Enhancements**  
   - Improve the \( f(m) \) approximation for higher accuracy.  
   - Explore using the **lognormal PDF equation** from Boudad and Cirelli to estimate the PBH formation mass instead of treating it as an input.

---

## Challenges

### Key Challenges  
- Mastering advanced numerical integration techniques.  
- Efficiently programming large-scale simulations.  
- Refining the \( f(m) \) approximation.

### Addressing Challenges  
- Weekly mentor meetings with Alexandra Klipfel in **26-541, Building 26** every Monday at **1 PM**.  
- Regular updates and discussions via email.

---

## Goals

The ultimate goal is to develop a robust algorithm for PBH mass evolution with reasonable accuracy. Achieving this will:
- Enhance understanding of **Hawking radiation** and its relationship to PBH mass.  
- Provide insights into **dark matter theories**.

This project offers a unique opportunity to deepen my understanding of **dark matter** and **black holes** while improving my coding and problem-solving skills. It aligns closely with my aspirations in astrophysics and contributes to ongoing research led by **Professor Peter Fisher** and **Alexandra Klipfel**.

---

## Acknowledgments

Special thanks to **Professor Peter Fisher** and **Alexandra Klipfel** for their guidance and mentorship on this project.
