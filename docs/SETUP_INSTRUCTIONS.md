# Setup Instructions

## Step-by-Step Installation Guide

### 1. Open PowerShell/Terminal
Open PowerShell or Command Prompt in your project directory.

### 2. Navigate to the Project Directory
```powershell
cd C:\Users\user\OneDrive\Desktop\projects\check_fonts
```

Or if you're already in the `projects` folder:
```powershell
cd check_fonts
```

### 3. Create a Virtual Environment (Recommended)
This keeps your project dependencies isolated:

```powershell
python -m venv venv
```

### 4. Activate the Virtual Environment
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# OR use Command Prompt style:
.\venv\Scripts\activate.bat
```

You should see `(venv)` appear at the start of your command prompt.

### 5. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 6. Verify Installation
```powershell
python check_fonts.py
```

If this runs without errors, you're all set!

---

## Quick Reference

**Current Directory Structure:**
```
projects/
└── check_fonts/
    ├── check_fonts.py
    ├── data_collection.py
    ├── requirements.txt
    └── venv/          (created after step 3)
```

**Always activate virtual environment before running scripts:**
```powershell
.\venv\Scripts\Activate.ps1
```

**To deactivate virtual environment:**
```powershell
deactivate
```

