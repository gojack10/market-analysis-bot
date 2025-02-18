# Market Analysis Bot Setup Guide

# [Video Tutorial Windows](https://youtu.be/YzpSC4xwdHs)
# [Video Tutorial Mac](https://youtu.be/hSXXVzDpsHk)

## [Skip to macOS Setup Section](#macos-setup-tutorial) | [Skip to Windows Setup Section](#windows-setup-tutorial)  
*(Basic Python explanation below)*

---

## Python Basics

Python is a programming language that lets you tell your computer what to do using easy-to-read instructions.

### Example: Adding Two Numbers

```python
number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: "))

sum_of_numbers = number1 + number2

print("The sum of", number1, "and", number2, "is", sum_of_numbers)
```

In Python, we use **packages** to avoid building complex functions from scratch. Packages are like sets of ready-made tools that we add to our project. Sometimes, we update and change the tools we use for this bot, and different versions of tools can conflict or be outdated. This is why it’s very, very, very good practice to use virtual environments.

A **factorial** of a non-negative integer $n$, written as $n!$, is the product of all positive integers less than or equal to $n$. For example, $5! = 5 \times 4 \times 3 \times 2 \times 1 = 120$.

### Example: Calculating Factorial Manually

```python
def factorial(number):
    if number == 0:
        return 1
    result = 1
    for i in range(1, number + 1):
        result *= i
    return result

number = 5
print("The factorial of", number, "is:", factorial(number))
```

### Example: Calculating Factorial Using the `math` Package

```python
import math

number = 5
print("The factorial of", number, "is:", math.factorial(number))
```

**Virtual environments** are like having separate rooms for different projects in your house. When a project uses many different packages, you don’t want the tools from one project to mix with or conflict with another. By setting up a separate environment, each project gets its own “room” where only the specific tools it needs are available. This makes it much easier to manage updates and changes without breaking the project.

Python package managers are tools that help you easily grab, create, and manage the tools and environments (or packages and virtual environments) we need for our projects. **UV** is a new, super-fast package manager that combines the features of pip, pipx, Poetry, and virtualenv into one, making it simpler and quicker to create isolated project environments and manage dependencies efficiently.

---

## macOS Setup Tutorial
<a name="macos-setup-tutorial"></a>

On macOS, **brew** is another package manager for your whole computer that makes installing and managing software as simple as typing commands in your Terminal. Instead of manually downloading installer files, brew handles everything for you, which is especially handy for keeping your software up-to-date.

### **Step 1: Install Brew and Required Software**

1. **Install Brew:**  
   Open your Terminal and paste the following command:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.13:**  
   Once brew is installed, use it to install Python 3.13 (the version we use for our project) by running:
   ```bash
   brew install python@3.13
   ```

3. **Install UV:**  
   Next, install **UV**, our fast package manager, with:
   ```bash
   brew install uv
   ```

4. **Install TA-Lib:**  
   Next, install **TA-Lib**, with:
   ```bash
   brew install ta-lib
   ```

---

### **Step 2: Set Up Your Project**

Before setting up your project, it's useful to understand the **root directory**. The root directory is the top-level folder on your computer where all other folders and files are stored—it’s like the main folder that holds everything. This is where your terminal defaults to when it first opens.

4. **Download and Unzip the Project:**

- **Option A:**  
  Download the project folder directly from GitHub:
  1. Go to the [project repository](https://github.com/gojack10/market-analysis-bot) on GitHub.
  2. Click the **Code** button and select **Download ZIP**.
  3. Once downloaded, unzip the folder to your desired location on your computer.
  4. **Navigate to the Project Folder:**  
     After unzipping or cloning, change into the project directory:
     ```bash
     cd /saved-folder/market-analysis-bot-main
     ```

- **Option B:**  
  If you have GitHub CLI (or Git) set up, you can clone the repository directly:
  1. Open your terminal.
  2. Navigate to the folder where you want to place the project using:
     ```bash
     cd path/to/your/desired/folder
     ```
  3. Clone the repository by running:
     ```bash
     git clone https://github.com/gojack10/market-analysis-bot.git
     ```
  4. **Navigate to the Project Folder:**  
     After unzipping or cloning, change into the project directory:
     ```bash
     cd market-analysis-bot
     ```

---

### **Step 3: Prepare Your Project Environment**

6. **Create a Virtual Environment:**  
   Change into your project folder (if not already there) and run:
   ```bash
   uv venv -p 3.13
   ```
   This command creates a virtual environment using Python 3.13 to keep your project’s dependencies isolated.

7. **Activate the Virtual Environment:**  
   Activate your virtual environment with:
   ```bash
   source .venv/bin/activate
   ```
   This ensures that any Python packages you install are used only by this project.

8. **Install Project Dependencies:**  
   Finally, install the necessary packages listed in the requirements file:
   ```bash
   uv pip install -r requirements.txt
   ```

---

### **Step 4: Test Your Setup**

Now, you’re all set! Test your setup by running:
```bash
python gui3.py
```

---

Below is a Windows-adapted version of the tutorial. Note that Windows users typically use a package manager like [Chocolatey](https://chocolatey.org/) (or [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/) on Windows 10/11) instead of Homebrew. This guide will use Chocolatey. If you don’t already have Chocolatey installed, follow the instructions in **Step 1A** below.

---

## Windows Setup Tutorial
<a name="windows-setup-tutorial"></a>

On Windows, a package manager such as **Chocolatey** lets you install and manage software via command-line commands. This guide uses Chocolatey to install the required software, set up your project, and create an isolated Python environment.

---

### **Step 1: Install Chocolatey and Required Software**

#### **Step 1A: Install Chocolatey (if not already installed)**
1. **Open PowerShell as Administrator:**  
   - Click the Start button, type “PowerShell”, right-click **Windows PowerShell**, and select **Run as administrator**.
2. **Install Chocolatey:**  
   Paste the following command into your PowerShell window and press Enter:
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force; `
     [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
     iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```
   Follow any prompts to complete the installation. Once installed, close and reopen your PowerShell window (you no longer need administrator privileges for the following steps).

#### **Step 1B: Install Python 3.13**
1. **Install Python:**  
   Use Chocolatey to install Python 3.13 by running:
   ```powershell
   choco install python313
   ```
   > **Note:** If the exact version isn’t available via Chocolatey, you can download the installer from [python.org](https://www.python.org/downloads/) and install Python 3.13 manually. Make sure to check the box to “Add Python to PATH” during installation.

#### **Step 1C: Install UV**
2. **Install UV (our fast package manager):**  
   If UV is available on Chocolatey, run:
   ```powershell
   choco install uv
   ```
   If it’s not available via Chocolatey, please follow the project’s [installation instructions](https://github.com/gojack10/market-analysis-bot) for UV on Windows.

#### **Step 1D: Install TA-Lib**
3. **Install TA-Lib:**  

On Windows, installing TA-Lib requires downloading the .msi executable installer from the official TA-Lib website. Follow these steps:

1. **Download the Installer:**  
   Visit the [TA-Lib Installation Page](https://ta-lib.org/install/) and download the .msi executable installer.

2. **Run the Installer:**  
   Execute the downloaded installer and follow the on-screen instructions to install TA-Lib on your system.

---

### **Step 2: Set Up Your Project**

Before setting up your project, it’s useful to understand the **root directory**. On Windows, this is typically the main folder where your user files reside (for example, `C:\Users\YourName`). When you open a new PowerShell or Command Prompt window, it defaults to your user folder.

4. **Download and Unzip the Project:**

- **Option A: Download ZIP from GitHub**
  1. Visit the [project repository](https://github.com/gojack10/market-analysis-bot) on GitHub.
  2. Click the **Code** button and select **Download ZIP**.
  3. Once downloaded, right-click the ZIP file and choose **Extract All…** to unzip it to your desired location.
  4. **Navigate to the Project Folder:**  
     Open PowerShell (or Command Prompt) and change into the project directory. For example:
     ```powershell
     cd C:\path\to\your\unzipped\market-analysis-bot-main
     ```
- Alternatively (and easier) just go to the folder your downloaded everything to and just type `powershell` in the address bar

- **Option B: Clone the Repository using Git**
  1. If you have Git installed, open PowerShell.
  2. Navigate to your chosen folder:
     ```powershell
     cd C:\path\to\your\desired\folder
     ```
  3. Clone the repository:
     ```powershell
     git clone https://github.com/gojack10/market-analysis-bot.git
     ```
  4. **Navigate to the Project Folder:**
     ```powershell
     cd market-analysis-bot
     ```

---

### **Step 3: Prepare Your Project Environment**

5. **Create a Virtual Environment:**  
   In your project folder, run:
   ```powershell
   uv venv -p 3.13
   ```
   This command creates a virtual environment using Python 3.13, keeping your project’s dependencies isolated.

6. **Activate the Virtual Environment:**  
   Run:
   ```powershell
   .venv\Scripts\activate
   ```
   Once activated, your prompt should show the name of your virtual environment in parenthesis.

---

7. **Install Project Dependencies:**  
   With the virtual environment active, install the necessary packages by running:
   ```powershell
   uv pip install -r requirements.txt
   ```

---

#### Troubleshooting: Visual Studio Build Tools & Missing Packages

Sometimes you might encounter errors during installation or when running your project. For example, you may see an error like:

```powershell
[stderr]
C:\Users\kashk\AppData\Local\uv\cache\builds-v0\.tmpqbWvfm\Lib\site-packages\setuptools\config\_apply_pyprojecttoml.py:81:
SetuptoolsWarning: `install_requires` overwritten in `pyproject.toml` (dependencies)
  corresp(dist, value, root_dir)
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools":
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

If you see the above error, follow these steps:

1. **Install Visual Studio Build Tools:**
   - Visit: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Download and run `vs_BuildTools.exe`.
   - In the installer, **make sure "Desktop development with C++" is checked**, then click **Install**.
   
---

### **Step 4: Test Your Setup**

8. **Run the Application:**  
   Test your configuration by launching the project:
   ```powershell
   python gui3.py
   ```
