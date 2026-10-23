# Environment Setup & Scikit-learn Installation Guide

## A. Topic Introduction
- **What is an Environment Setup?** Machine Learning code likhne aur chalane ke liye humein apne computer par **Python** programming language aur kuch special scientific libraries (`scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`) install karni hoti hain.
- **Simple Words Mein:** Jaise khana pakane se pehle kitchen mein choolha, bartan aur masalay (ingredients) set kiye jate hain, bilkul waise hi coding se pehle software tools install karna **Environment Setup** kehlata hai.
- **Technical Definition:** Configuring an isolated Python runtime environment (`venv` or `conda`) with required third-party numerical and machine learning packages via a package manager (`pip` or `conda`).

## B. Why Do We Need It?
- **Problem Solved:** Alag alag projects ko Python aur libraries ke mukhtalif versions chahiye hote hain. Agar hum sab kuch global system par install kar dein to version conflicts ("dependency hell") paida ho jate hain.
- **Importance:** **Virtual Environment (`venv`)** har project ko ek alag kamra (isolated folder) deta hai taake ek project ki installation doosre project ko kharab na kare.
- **Without It:** Aap ka purana code achanak chalna band ho sakta hai jab aap kisi naye project ke liye library update karein.

## C. Supported Python Environments & System Requirements
Official [Scikit-learn Installation Documentation](https://scikit-learn.org/stable/install.html) ke mutabiq:
- **Supported Python Versions:** Scikit-learn ke modern versions (v1.5 se v1.9+) ke liye **Python 3.9, 3.10, 3.11, 3.12, ya 3.13** recommended hai.
- **Operating Systems:** Windows (10/11), macOS, aur Linux (Ubuntu/Debian/Fedora) sab par mukammal support mojood hai.
- **RAM & Hardware:** Seekhne ke liye 4 GB RAM (8 GB recommended) aur koi bhi normal dual-core/quad-core processor kafi hai. Scikit-learn CPU par chalta hai, is ke liye mehangay GPU (Graphics Card) ki zaroorat nahi hoti!

## D. Real-World Analogy
Socho aap ke ghar mein ek hi almari hai jismein bijli ke auzar, kitchen ke masalay, aur school ki kitabein sab ek sath phaink di gayi hain—jab koi cheez dhoondni ho to sab ulajh jata hai. Is ke bar-aks **Virtual Environment (`venv`)** har kaam ke liye alag daraaz (drawer) banane jaisa hai: "ML Course" ka apna daraaz hai jismein sirf zaroori tools saaf suthray rakhay hain.

## E. Step-by-Step Installation Commands

### Step 1: Python Install Karein
1. [python.org/downloads](https://www.python.org/downloads/) par jayein aur Python 3.11 ya 3.12 download karein.
2. **Windows Users ke liye Bohat Zaroori Note:** Installer chalate waqt pehli screen par neeche **"Add python.exe to PATH"** wale checkbox par LAZMI tick lagayein, phir "Install Now" par click karein.

### Step 2: Project Folder aur Virtual Environment (`venv`) Banayein
Apna Terminal (macOS/Linux) ya Command Prompt / PowerShell (Windows) kholein aur yeh commands likhein:

```bash
# 1. Course ke liye naya folder banayein aur us ke andar jayein
mkdir sklearn_course
cd sklearn_course

# 2. Virtual environment banayein (naam: .venv)
python -m venv .venv
```

### Step 3: Virtual Environment Activate Karein
- **Windows (Command Prompt):**
  ```cmd
  .venv\\Scripts\\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\\Scripts\\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
Jab environment activate ho jayega, to aap ki command line ke shuru mein `(.venv)` likha nazar aayega.

### Step 4: Scikit-learn aur Companion Libraries Install Karein
Ab `pip` ke zariye tamam zaroori packages ek sath install karein:

```bash
pip install --upgrade pip
pip install -U scikit-learn numpy pandas matplotlib seaborn joblib jupyterlab
```

## F. Complete Working Verification Code
Installation mukammal hone ke baad, ek file `verify_setup.py` banayein ya Jupyter Notebook mein yeh code run karein:

```python
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import joblib

print("=== Environment Verification Report ===")
print(f"Python Version      : {sys.version.split()[0]}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"NumPy Version       : {np.__version__}")
print(f"Pandas Version      : {pd.__version__}")
print(f"Matplotlib Version  : {matplotlib.__version__}")
print(f"Seaborn Version     : {sns.__version__}")
print(f"Joblib Version      : {joblib.__version__}")

# Detailed build and dependency check
sklearn.show_versions()
```

## G. Code Explanation (Line-by-Line in Roman Urdu)
1. `import sys`: Python ke system module ko import karta hai taake hum Python ka version check kar saken.
2. `import sklearn`: Scikit-learn library ko import karta hai. **Dhyan dein:** Install karte waqt hum `pip install scikit-learn` likhte hain, lekin Python code ke andar hamesha `import sklearn` likha jata hai!
3. `import numpy as np` aur `import pandas as pd`: NumPy aur Pandas ko standard short names (`np` aur `pd`) ke sath load karta hai.
4. `print(f"Scikit-learn Version: {sklearn.__version__}")`: Har library ka `__version__` attribute print kar ke confirm karta hai ke sab theek install hua hai.
5. `sklearn.show_versions()`: Yeh Scikit-learn ka built-in diagnostic function hai jo OS, Python, BLAS/LAPACK math libraries, aur dependencies ki mukammal tafseel dikhata hai.

## H. Expected Output
```text
=== Environment Verification Report ===
Python Version      : 3.12.4
Scikit-learn Version: 1.6.1
NumPy Version       : 2.1.2
Pandas Version      : 2.2.3
Matplotlib Version  : 3.9.2
Seaborn Version     : 0.13.2
Joblib Version      : 1.4.2

System:
    python: 3.12.4 ...
Python dependencies:
    sklearn: 1.6.1
    pip: 24.2
    numpy: 2.1.2
    scipy: 1.14.1
...
```
*(Note: Versions aap ke install karne ke waqt ke hisab se thore naye ya mukhtalif ho sakte hain, maslan Scikit-learn 1.5.x se 1.9.x tak).*

## I. Common Installation Errors & Fixes
1. **`ModuleNotFoundError: No module named 'sklearn'`**
   - **Wajah:** Aap ne `scikit-learn` install nahi kiya, ya phir install kisi aur environment mein kiya hai aur Jupyter Notebook kisi doosre Python kernel par chal raha hai.
   - **Hal (Fix):** Terminal mein confirm karein ke `(.venv)` active hai aur `python -m pip install scikit-learn` chalayein. Jupyter ke andar cell mein `%pip install scikit-learn` likh kar run karein.
2. **`pip install sklearn` Error**
   - **Wajah:** Purane waqton mein log `pip install sklearn` likhte thay jo ab **deprecated/removed** hai.
   - **Hal (Fix):** Hamesha `pip install scikit-learn` likhein.
3. **Windows PowerShell ExecutionPolicy Error**
   - **Wajah:** Windows PowerShell security ki wajah se `.ps1` scripts ko block karta hai.
   - **Hal (Fix):** PowerShell mein `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` run karein, ya simple `cmd` (Command Prompt) use karein.

## J. Quick Revision
- Installation package ka naam `scikit-learn` hai, jabke Python import ka naam `sklearn` hai.
- Hamesha har project ke liye `python -m venv .venv` ke zariye virtual environment banayein.
- `sklearn.__version__` se apna installed version check karein.

---
**Next Chapter:** [01_AI_AND_MACHINE_LEARNING.md](01_AI_AND_MACHINE_LEARNING.md) ya [PYTHON_PREREQUISITES.md](PYTHON_PREREQUISITES.md)
