# EasyHipPlanner
# 🦴 Easy Hip Planner

![License](https://img.shields.io/badge/License-Non--Commercial-red.svg)
![Medical Use](https://img.shields.io/badge/Medical%20Use-PROHIBITED-red.svg)
![3D Slicer](https://img.shields.io/badge/3D%20Slicer-5.0%2B-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%7CmacOS%7CLinux-lightgrey.svg)

⚠️ **NON-COMMERCIAL USE ONLY** ⚠️  
⚠️ **NOT FOR MEDICAL USE** ⚠️

Educational tool for hip replacement planning in 3D Slicer.

## 🚀 Features

- **X-ray and CT DICOM import**
- **Anatomical landmark placement** (Cups, Trochanters)
- **Femoral head positioning calculations**
- **Reference line measurements**
- **STL implant import and scaling**
- **Magnification correction tools**
- **Rotational correction**

## 📋 Requirements

- 3D Slicer 5.0 or newer
- Python environment with VTK support

## 🔧 Installation

1. **Download** the `EasyHipPlanner.py` file from this repository
2. **Open 3D Slicer**
3. **Go to:** Python console
4. **Paste:** `EasyHipPlanner.py`
5. **Use**


## 📖 Quick Start

1. **Import X-ray DICOM** image using "📁 X-ray" button
2. **Press "Show All"** button to display all anatomical elements
3. **Place Cups** in hip joint sockets according to L/R side
4. **Place Tr** on trochanter peaks (greater trochanters recommended)
5. **Adjust Ischial line** position to pelvis inclination
6. **Select operated side** and click "Show/Hide Head"
7. **Click "Toggle Upper Reference Line"**
8. **Read the "Head-Tr distance"** result for positioning guidance

## 📝 Important Notes

- **⚠️ EDUCATIONAL PURPOSE ONLY** - Not for clinical use
- The "Head-Tr distance" shows where the endoprosthesis head should be relative to your chosen trochanter point
- Intraoperatively, you can use K-wire through the trochanter peak to assess relative position
- If image turns into strip after scene reset, click "X-ray mode" to restore view
- Double-click in 2D view enables/disables manipulators

## 📧 Contact

- **Email:** przemek.czuma@gmail.com
- **Website:** [inteligencja.org.pl](https://inteligencja.org.pl)
- **LinkedIn:** [linkedin.com/in/inteligencjawzdrowiu](https://www.linkedin.com/in/inteligencjawzdrowiu/)

## 📄 License

This project is licensed under a **Non-Commercial License** - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- ❌ Not intended for clinical use
- ❌ Not intended for medical diagnosis  
- ❌ Not intended for patient treatment
- ❌ Not validated for medical applications

Use at your own risk. The author disclaims all responsibility for any medical decisions or outcomes.
