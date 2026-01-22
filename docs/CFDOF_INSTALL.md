# CfdOF / OpenFOAM Installation Guide

CFD simulation for microfluidics (PHLoC organoid culture, brain glymphatics).

## macOS Installation

### 1. Install OpenFOAM via Homebrew

```bash
# Add the OpenFOAM tap
brew tap gerlero/openfoam

# Install OpenFOAM (downloads ~1.1GB app)
brew install --cask openfoam
```

This installs OpenFOAM as `/Applications/OpenFOAM-v2512.app`.

### 2. Mount OpenFOAM Volume

The macOS version uses a disk image. Mount it before use:

```bash
# Mount the volume
/Applications/OpenFOAM-v2512.app/Contents/Resources/volume mount

# Verify it's mounted
ls /Volumes/OpenFOAM-v2512/etc/bashrc
```

The volume mounts at `/Volumes/OpenFOAM-v2512`.

### 3. Test OpenFOAM

```bash
# Run a solver through the wrapper
/opt/homebrew/bin/openfoam simpleFoam -help

# Or use the shortcut (after brew install)
openfoam simpleFoam -help
```

### 4. Install CfdOF in FreeCAD

1. Open FreeCAD
2. Go to **Tools → Addon Manager**
3. Search for "CfdOF"
4. Click **Install**
5. Restart FreeCAD

### 5. Configure CfdOF

In FreeCAD, CfdOF needs to know where OpenFOAM is:

**Option A: Via GUI**
1. Go to **Edit → Preferences → CfdOF**
2. Set Installation Path to `/Volumes/OpenFOAM-v2512`
3. Click "Run dependency checker"

**Option B: Via Python (or MCP)**
```python
import FreeCAD
prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/CfdOF")
prefs.SetString("InstallationPath", "/Volumes/OpenFOAM-v2512")
```

### 6. Install Optional Dependencies

```bash
# ParaView for visualization (optional)
brew install --cask paraview

# Gmsh is bundled with FreeCAD, but standalone:
brew install gmsh
```

## Linux Installation

```bash
# Ubuntu/Debian - OpenFOAM from official repo
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt install openfoam11

# Source OpenFOAM environment
source /opt/openfoam11/etc/bashrc

# Install CfdOF via FreeCAD Addon Manager (same as macOS)
```

## Verify Installation

Run from Python:

```python
import xmlrpc.client
server = xmlrpc.client.ServerProxy('http://localhost:9875', allow_none=True)

# Check CfdOF configuration
result = server.execute_code('''
import CfdOF.CfdTools as CfdTools
runtime = CfdTools.getFoamRuntime()
foam_dir = CfdTools.getFoamDir()
print(f"Runtime: {runtime}")
print(f"OpenFOAM: {foam_dir}")
''')
print(result)
```

Expected output:
```
Runtime: Posix
OpenFOAM: /Volumes/OpenFOAM-v2512
```

## Troubleshooting

### "OpenFOAM installation path not set"

Mount the volume first:
```bash
/Applications/OpenFOAM-v2512.app/Contents/Resources/volume mount
```

### "Directory is not a recognised OpenFOAM installation"

CfdOF expects `etc/bashrc` in the installation path. Use the mounted volume path, not the app bundle:
- Wrong: `/Applications/OpenFOAM-v2512.app`
- Correct: `/Volumes/OpenFOAM-v2512`

### FreeCAD RPC not responding

Start the RPC server in FreeCAD:
1. Install MCP addon (from `../freecad-mcp`)
2. Click **MCP Addon toolbar → Start RPC Server**
3. Server runs on `localhost:9875`

## Usage

See `examples/cfd_microfluidics_example.py` for complete usage.

Quick test via twosphere MCP:

```python
# PHLoC organoid culture
result = await handle_cfd_microfluidics({
    'geometry_type': 'phloc',
    'channel_diameter_um': 10,
    'velocity_um_s': 10000,
    'fluid': 'water'
})

# Brain glymphatic flow
result = await handle_cfd_microfluidics({
    'geometry_type': 'glymphatic',
    'channel_diameter_um': 20,
    'velocity_um_s': 10,
    'fluid': 'csf'
})
```

## References

- [CfdOF Workbench Wiki](https://wiki.freecad.org/CfdOF_Workbench)
- [OpenFOAM macOS App](https://github.com/gerlero/openfoam-app)
- [OpenFOAM Documentation](https://www.openfoam.com/documentation)
