#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Project Configuration Wizard - Simplified Version
Focus on generating build scripts with hardcoded paths
"""

import os
import sys
import json
import platform
import subprocess
import shutil
import re
from pathlib import Path

# Set output encoding to UTF-8 for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# --- Terminal Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    OK = '\033[92m'  # Same as GREEN for success messages
    WARN = '\033[93m'
    INFO = '\033[96m'  # Using CYAN color for INFO
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- Project Dependency Requirements ---
class Requirements:
    CMAKE_MIN = "3.24.0"
    PYTHON_MIN = "3.11"
    NUMPY_VERSION = "2.3.4"
    GCC_VERSION = "13"
    MSVC_VERSION = "14.44"  # Corresponds to cl.exe 14.44.35207
    CUDA_VERSION = "12.8"
    CUDNN_VERSION = "8"

    @staticmethod
    def version_compare(v1: str, v2: str) -> int:
        """比较版本号 v1 >= v2 返回 >=0, v1 < v2 返回 -1"""
        def version_key(v):
            return [int(x) for x in re.findall(r'\d+', v)]

        v1_parts = version_key(v1)
        v2_parts = version_key(v2)

        # 补齐长度
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))

        for v1_part, v2_part in zip(v1_parts, v2_parts):
            if v1_part > v2_part:
                return 1
            elif v1_part < v2_part:
                return -1

        return 0

class SmartConfigurator:
    def __init__(self):
        self.system = platform.system()
        self.config = {}
        self.config_file = Path("config/project_config.json")
        self.cmake_cache = Path("config/user_paths.cmake")

    def check_basic_tools(self) -> bool:
        """Check basic build tools"""
        print(f"{Colors.BLUE}[Step 1/7] Checking basic build tools...{Colors.ENDC}")

        # Check CMake
        cmake_ok = self.check_cmake()

        # Check Ninja
        ninja_ok = self.check_ninja()

        # Check vcpkg
        vcpkg_ok = self.check_vcpkg()

        if not cmake_ok or not ninja_ok or not vcpkg_ok:
            print(f"{Colors.FAIL}Missing required build tools!{Colors.ENDC}")
            self.suggest_tool_installation(cmake_ok, ninja_ok, vcpkg_ok)
            return False

        return True

    def check_cmake(self) -> bool:
        """Check CMake version"""
        try:
            result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_match = re.search(r'cmake version (\d+\.\d+\.\d+)', result.stdout)
                if version_match:
                    version = version_match.group(1)
                    if Requirements.version_compare(version, Requirements.CMAKE_MIN) >= 0:
                        print(f"  {Colors.GREEN}[OK] CMake {version}{Colors.ENDC}")
                        self.config['cmake_version'] = version
                        return True
                    else:
                        print(f"  {Colors.WARN}[ERROR] CMake version too low: {version} < {Requirements.CMAKE_MIN}{Colors.ENDC}")
        except FileNotFoundError:
            print(f"  {Colors.FAIL}[ERROR] CMake not found{Colors.ENDC}")
        return False

    def check_ninja(self) -> bool:
        """Check Ninja with intelligent search including vcpkg"""
        # 1. First check ninja in PATH
        if shutil.which('ninja'):
            try:
                result = subprocess.run(['ninja', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    print(f"  {Colors.GREEN}[OK] Ninja {version} (PATH){Colors.ENDC}")
                    self.config['ninja_path'] = 'ninja'
                    return True
            except:
                pass

        # 2. Check if vcpkg has ninja installed
        if 'vcpkg_root' in self.config:
            vcpkg_ninja = Path(self.config['vcpkg_root']) / "installed" / "x64-windows" / "tools" / "ninja" / "ninja.exe"
            if vcpkg_ninja.exists():
                try:
                    result = subprocess.run([str(vcpkg_ninja), '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        print(f"  {Colors.GREEN}[OK] Ninja {version} (vcpkg){Colors.ENDC}")
                        self.config['ninja_path'] = str(vcpkg_ninja)
                        return True
                except:
                    pass
            else:
                # Suggest installing ninja via vcpkg
                print(f"  {Colors.INFO}vcpkg found but ninja not installed. You can install it with: vcpkg install ninja{Colors.ENDC}")

        # 3. Smart search in other locations
        print(f"  {Colors.INFO}Searching for Ninja installations...{Colors.ENDC}")
        ninja_installations = self.find_ninja_installations()

        if ninja_installations:
            # Select the first valid installation
            for ninja_exe in ninja_installations[:3]:  # Show max 3 options
                try:
                    result = subprocess.run([str(ninja_exe), '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        print(f"  {Colors.GREEN}[OK] Ninja {version}: {ninja_exe}{Colors.ENDC}")
                        self.config['ninja_path'] = str(ninja_exe)
                        return True
                except:
                    continue

        # 4. If ninja not found and vcpkg is available, suggest installation
        if 'vcpkg_root' in self.config:
            print(f"  {Colors.WARN}[ERROR] Ninja not found, but vcpkg is available{Colors.ENDC}")
            print(f"  {Colors.CYAN}You can install ninja with: vcpkg install ninja{Colors.ENDC}")
            return False

        # 5. Prompt user
        return self.prompt_ninja_path()

    def find_ninja_installations(self) -> list:
        """Intelligently find Ninja installations"""
        ninja_paths = []

        if self.system == "Windows":
            # Windows common installation paths
            search_paths = [
                Path.home() / "AppData/Local/Programs",
                Path(os.environ.get("PROGRAMFILES", "C:/Program Files")),
                Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")),
                Path("C:/tools"),
                Path("D:/tools"),
                Path("T:/Softwares"),
            ]

            # Python related ninja
            python_paths = []
            try:
                python_dir = Path(sys.executable).parent
                python_paths = [
                    python_dir / "Scripts",
                    python_dir,
                ]
            except:
                pass
            search_paths.extend(python_paths)

            # Search for ninja.exe
            for base_path in search_paths:
                if base_path.exists():
                    try:
                        # Recursive search (limit depth to avoid slowness)
                        for ninja_exe in base_path.rglob("ninja.exe"):
                            if ninja_exe.exists():
                                ninja_paths.append(ninja_exe)
                    except:
                        continue

            # Check ninja in vcpkg (if vcpkg is already found)
            if 'vcpkg_root' in self.config:
                vcpkg_root = self.config['vcpkg_root']
                vcpkg_ninja = Path(vcpkg_root) / "installed" / "x64-windows" / "tools" / "ninja" / "ninja.exe"
                if vcpkg_ninja.exists():
                    ninja_paths.append(vcpkg_ninja)

        else:
            # Linux common installation paths
            search_paths = [
                Path("/usr/bin"),
                Path("/usr/local/bin"),
                Path.home() / ".local/bin",
                Path("/opt"),
            ]

            for base_path in search_paths:
                ninja_path = base_path / "ninja"
                if ninja_path.exists() and os.access(ninja_path, os.X_OK):
                    ninja_paths.append(ninja_path)

        return ninja_paths

    def prompt_ninja_path(self) -> bool:
        """Prompt user for Ninja path"""
        print(f"  {Colors.WARN}[ERROR] Ninja not found{Colors.ENDC}")
        print(f"  {Colors.CYAN}Ninja is a required build tool{Colors.ENDC}")

        while True:
            if self.system == "Windows":
                path_input = input(f"  {Colors.CYAN}Enter full path to ninja.exe (or press Enter to skip): {Colors.ENDC}").strip().replace('"', '')
            else:
                path_input = input(f"  {Colors.CYAN}Enter full path to ninja (or press Enter to skip): {Colors.ENDC}").strip().replace('"', '')

            if not path_input:
                print(f"  {Colors.WARN}Skipping Ninja configuration, may affect build{Colors.ENDC}")
                return False

            ninja_path = Path(path_input)

            # Normalize path separators
            if self.system == "Windows":
                ninja_path = Path(str(ninja_path).replace('/', '\\'))
                ninja_exe = ninja_path / "ninja.exe" if ninja_path.is_dir() else ninja_path
            else:
                ninja_path = Path(str(ninja_path).replace('\\', '/'))
                ninja_exe = ninja_path / "ninja" if ninja_path.is_dir() else ninja_path

            # Verify path
            if ninja_exe.exists():
                if self.system == "Windows" or os.access(ninja_exe, os.X_OK):
                    try:
                        result = subprocess.run([str(ninja_exe), '--version'], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            version = result.stdout.strip()
                            print(f"  {Colors.GREEN}[OK] Ninja {version}: {ninja_exe}{Colors.ENDC}")
                            self.config['ninja_path'] = str(ninja_exe)
                            return True
                    except:
                        pass

                print(f"  {Colors.FAIL}[ERROR] Invalid Ninja executable: {ninja_exe}{Colors.ENDC}")
            else:
                print(f"  {Colors.FAIL}[ERROR] File does not exist: {ninja_exe}{Colors.ENDC}")

    def check_vcpkg(self) -> bool:
        """Check vcpkg using intelligent search"""
        # 1. Check VCPKG_ROOT environment variable first
        vcpkg_env = os.environ.get('VCPKG_ROOT')
        if vcpkg_env and Path(vcpkg_env).exists():
            print(f"  {Colors.GREEN}[OK] vcpkg (VCPKG_ROOT): {vcpkg_env}{Colors.ENDC}")
            self.config['vcpkg_root'] = vcpkg_env
            return True

        # 2. Check PATH for vcpkg.exe
        vcpkg_in_path = shutil.which('vcpkg')
        if vcpkg_in_path:
            vcpkg_path = Path(vcpkg_in_path).parent.parent
            if (vcpkg_path / '.vcpkg-root').exists():
                print(f"  {Colors.GREEN}[OK] vcpkg (PATH): {vcpkg_path}{Colors.ENDC}")
                self.config['vcpkg_root'] = str(vcpkg_path)
                return True

        # 3. Check common installation locations
        print(f"  {Colors.INFO}Searching for vcpkg in common locations...{Colors.ENDC}")
        if self.system == "Windows":
            common_paths = [
                Path("C:/vcpkg"),
                Path("D:/vcpkg"),
                Path("T:/vcpkg"),
                Path("T:/Softwares/vcpkg"),
                Path.home() / "vcpkg",
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "vcpkg",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "vcpkg",
            ]
        else:
            common_paths = [
                Path("/opt/vcpkg"),
                Path("/usr/local/vcpkg"),
                Path.home() / "vcpkg",
            ]

        # Check current directory
        local_vcpkg = Path('vcpkg')
        if local_vcpkg.exists() and (local_vcpkg / '.vcpkg-root').exists():
            print(f"  {Colors.GREEN}[OK] vcpkg (local): {local_vcpkg.absolute()}{Colors.ENDC}")
            self.config['vcpkg_root'] = str(local_vcpkg.absolute())
            return True

        # Search common locations
        for vcpkg_path in common_paths:
            if vcpkg_path.exists() and (vcpkg_path / '.vcpkg-root').exists():
                print(f"  {Colors.GREEN}[OK] vcpkg: {vcpkg_path}{Colors.ENDC}")
                self.config['vcpkg_root'] = str(vcpkg_path)
                return True

        print(f"  {Colors.WARN}[ERROR] vcpkg not found{Colors.ENDC}")
        return False

    def suggest_tool_installation(self, cmake_ok: bool, ninja_ok: bool, vcpkg_ok: bool):
        """Suggest tool installation"""
        print(f"\n{Colors.CYAN}Installation suggestions:{Colors.ENDC}")

        if not cmake_ok:
            print(f"  - CMake: Download from https://cmake.org/download/")
            print(f"  - Or use package manager: brew install cmake (macOS), sudo apt install cmake (Ubuntu)")

        if not ninja_ok:
            print(f"  - Ninja: pip install ninja or sudo apt install ninja-build")

        if not vcpkg_ok:
            print(f"  - vcpkg:")
            print(f"    git clone https://github.com/microsoft/vcpkg.git")
            print(f"    cd vcpkg && .\\bootstrap-vcpkg.bat (Windows) or ./bootstrap-vcpkg.sh (Linux)")
            print(f"    export VCPKG_ROOT=/path/to/vcpkg")

    def setup_compiler(self) -> bool:
        """Setup compiler using intelligent search"""
        print(f"\n{Colors.BLUE}[Step 2/7] Setting up compiler...{Colors.ENDC}")

        if self.system == "Windows":
            # Initialize compilers configuration
            self.config['compilers'] = {}

            # Find MSVC
            msvc_info = self.find_msvc()
            if msvc_info:
                self.config['compilers']['msvc'] = msvc_info

            # Find MSYS2 GCC
            msys2_info = self.find_msys2_gcc()
            if msys2_info:
                self.config['compilers']['msys2'] = msys2_info

            # If at least one compiler found, set as primary for backward compatibility
            if 'msvc' in self.config['compilers']:
                self.config['compiler'] = self.config['compilers']['msvc']
            elif 'msys2' in self.config['compilers']:
                self.config['compiler'] = self.config['compilers']['msys2']

            # If at least one compiler found, return success
            return 'compilers' in self.config and self.config['compilers']
        else:
            print(f"  {Colors.INFO}Setting up Linux GCC...{Colors.ENDC}")
            return self.setup_linux_compiler()

    def setup_linux_compiler(self) -> bool:
        """Setup Linux GCC compiler"""
        gcc_info = self.find_linux_gcc()
        if gcc_info:
            self.config['compiler'] = gcc_info
            self.config['compilers'] = {'gcc': gcc_info}
            return True
        else:
            print(f"  {Colors.WARN}Linux GCC not found{Colors.ENDC}")
            return False

    def find_linux_gcc(self):
        """Find Linux GCC using intelligent search"""
        print(f"  [INFO] Looking for Linux GCC...")

        # 1. Check PATH for GCC
        gcc_info = self.find_gcc_in_path()
        if gcc_info:
            print(f"    {Colors.OK}Found GCC in PATH: {gcc_info['gcc_path']}{Colors.ENDC}")
            return gcc_info

        # 2. Try standard locations
        gcc_info = self.find_gcc_manual()
        if gcc_info:
            return gcc_info

        print(f"    [WARN] GCC not found")
        return None

    def find_gcc_in_path(self):
        """Find GCC in system PATH"""
        try:
            # Check which gcc is available
            result = subprocess.run(['which', 'gcc'], capture_output=True, text=True)
            if result.returncode == 0:
                gcc_path = result.stdout.strip()

                # Get version
                version_result = subprocess.run([gcc_path, '--version'], capture_output=True, text=True)
                version = None
                if version_result.returncode == 0:
                    # Parse version from output like "gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0"
                    version_match = re.search(r'gcc.*?\b(\d+\.\d+\.\d+)\b', version_result.stdout)
                    if version_match:
                        version = version_match.group(1)

                return {
                    'gcc_path': gcc_path,
                    'version': version,
                    'display_name': f'GCC {version}' if version else 'GCC'
                }
        except Exception as e:
            print(f"    [DEBUG] Failed to check gcc in PATH: {e}")
        return None

    def find_gcc_manual(self):
        """Find GCC in standard Linux locations"""
        potential_paths = [
            '/usr/bin/gcc',
            '/usr/local/bin/gcc',
            '/opt/gcc/bin/gcc',
            '/usr/bin/gcc-13',
            '/usr/bin/gcc-12',
            '/usr/bin/gcc-11'
        ]

        for gcc_path in potential_paths:
            try:
                if os.path.exists(gcc_path):
                    # Test if it works
                    result = subprocess.run([gcc_path, '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version_match = re.search(r'gcc.*?\b(\d+\.\d+\.\d+)\b', result.stdout)
                        version = version_match.group(1) if version_match else None

                        return {
                            'gcc_path': gcc_path,
                            'version': version,
                            'display_name': f'GCC {version}' if version else 'GCC'
                        }
            except Exception:
                continue

        return None

    def find_msvc(self):
        """Find MSVC using intelligent search"""
        print(f"  [INFO] Looking for MSVC...")

        # 1. Check PATH first
        path_cl = self.find_cl_in_path()
        if path_cl:
            print(f"    {Colors.OK}Found MSVC in PATH: {path_cl['cl_path']}{Colors.ENDC}")
            return path_cl

        # 2. Use vswhere
        vswhere_path = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if vswhere_path.exists():
            result = self.find_msvc_with_vswhere(vswhere_path)
            if result:
                return result

        # 3. Manual search
        return self.find_msvc_manual()

    def find_cl_in_path(self):
        """Find cl.exe in system PATH"""
        try:
            path_env = os.environ.get('PATH', '')
            path_dirs = path_env.split(os.pathsep)

            for path_dir in path_dirs:
                if not path_dir.strip():
                    continue
                path_dir = Path(path_dir.strip())
                path_str = str(path_dir).lower()

                # Check for Visual Studio related paths
                if any(keyword in path_str for keyword in ['visual studio', 'vc', 'msvc', 'microsoft visual c++']):
                    cl_path = path_dir / 'cl.exe'
                    if cl_path.exists():
                        # Extract version from path
                        version_match = re.search(r'MSVC\\(\d+\.\d+\.\d+)', path_str)
                        if version_match:
                            version = version_match.group(1) + ".0"
                            if Requirements.version_compare(version, Requirements.MSVC_VERSION) >= 0:
                                # Infer Visual Studio installation path
                                vs_path = path_dir
                                while len(vs_path.parts) > 2:
                                    parent_name = vs_path.parent.name.lower()
                                    if any(keyword in parent_name for keyword in ['visual studio', 'vs']):
                                        vs_path = vs_path.parent
                                        break
                                    vs_path = vs_path.parent

                                vcvars_path = vs_path / "VC/Auxiliary/Build/vcvars64.bat"
                                return {
                                    'cl_path': str(cl_path),
                                    'vcvars_path': str(vcvars_path) if vcvars_path.exists() else '',
                                    'version': version,
                                    'vs_path': str(vs_path)
                                }
        except Exception:
            pass
        return None

    def find_msvc_with_vswhere(self, vswhere_path):
        """Find MSVC using vswhere"""
        try:
            # Try different vswhere command combinations
            commands = [
                # Command 1: Latest VS with C++ workload
                [
                    str(vswhere_path),
                    "-latest",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-format", "json",
                    "-utf8"
                ],
                # Command 2: All VS instances with C++ workload
                [
                    str(vswhere_path),
                    "-all",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-format", "json",
                    "-utf8"
                ],
                # Command 3: All VS instances (fallback)
                [
                    str(vswhere_path),
                    "-latest",
                    "-format", "json",
                    "-utf8"
                ]
            ]

            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                      encoding='utf-8', errors='replace')

                if result.returncode == 0 and result.stdout:
                    installations = json.loads(result.stdout.strip())
                    if not installations:
                        continue

                    for vs_info in installations:
                        try:
                            vs_path = vs_info['installationPath']
                            vcvars_path = Path(vs_path) / "VC/Auxiliary/Build/vcvars64.bat"

                            if vcvars_path.exists():
                                # Find cl.exe path
                                cl_path = self.find_cl_executable(vs_path)
                                if not cl_path:
                                    print(f"    [WARN] cl.exe not found in VS installation")
                                    continue

                                # Get cl.exe version
                                cl_version = self.get_msvc_version(vcvars_path)
                                if cl_version and Requirements.version_compare(cl_version, Requirements.MSVC_VERSION) >= 0:
                                    print(f"    {Colors.OK}Found MSVC via vswhere: {vs_info.get('displayName', 'Unknown')}{Colors.ENDC}")
                                    return {
                                        'vs_path': vs_path,
                                        'vcvars_path': str(vcvars_path),
                                        'cl_path': str(cl_path),
                                        'version': cl_version,
                                        'vs_version': vs_info.get('installationVersion', 'unknown'),
                                        'display_name': vs_info.get('displayName', 'Unknown')
                                    }
                                else:
                                    print(f"    [DEBUG] MSVC found but version incompatible: {cl_version}")
                            else:
                                print(f"    [DEBUG] vcvars64.bat not found: {vcvars_path}")
                        except Exception as e:
                            print(f"    [DEBUG] Error processing VS installation: {e}")
                            continue
        except Exception as e:
            print(f"    [ERROR] vswhere failed: {e}")
        return None

    def find_cl_executable(self, vs_path):
        """Find cl.exe in Visual Studio installation"""
        try:
            # Common MSVC tool locations
            msvc_tools_path = Path(vs_path) / "VC/Tools/MSVC"
            if msvc_tools_path.exists():
                # Find the latest MSVC version
                version_dirs = [d for d in msvc_tools_path.iterdir() if d.is_dir()]
                if not version_dirs:
                    return None

                # Sort by version (latest first)
                version_dirs.sort(reverse=True)

                for version_dir in version_dirs:
                    cl_path = version_dir / "bin/Hostx64/x64/cl.exe"
                    if cl_path.exists():
                        return cl_path

            # Fallback: search recursively for cl.exe
            for cl_path in Path(vs_path).rglob("cl.exe"):
                if "bin" in str(cl_path) and "x64" in str(cl_path):
                    return cl_path

        except Exception as e:
            print(f"    [ERROR] Failed to find cl.exe: {e}")
        return None

    def find_msvc_manual(self):
        """Manual MSVC search as fallback"""
        common_paths = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio",
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio",
            Path("C:/Visual Studio"),
            Path("D:/Visual Studio"),
            Path("E:/Visual Studio"),
            Path("T:/Softwares/Visual Studio"),
        ]

        vs_years = ["2022", "2019", "2017"]
        editions = ["Community", "Professional", "Enterprise", "BuildTools"]

        for base_path in common_paths:
            if not base_path.exists():
                continue

            for year in vs_years:
                for edition in editions:
                    vs_path = base_path / year / edition
                    if vs_path.exists():
                        vcvars_path = vs_path / "VC/Auxiliary/Build/vcvars64.bat"
                        if vcvars_path.exists():
                            cl_version = self.get_msvc_version(vcvars_path)
                            if cl_version and Requirements.version_compare(cl_version, Requirements.MSVC_VERSION) >= 0:
                                return {
                                    'vs_path': str(vs_path),
                                    'vcvars_path': str(vcvars_path),
                                    'version': cl_version,
                                    'vs_version': f"{edition} {year}",
                                    'display_name': f"{edition} {year}"
                                }
        return None

    def get_msvc_version(self, vcvars_path):
        """Get MSVC cl.exe version"""
        try:
            cmd = f'"{vcvars_path}" && cl'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', timeout=10)

            # Look for version in stderr and stdout
            output_text = result.stderr + result.stdout
            if output_text:
                print(f"      [DEBUG] cl version output (first 200 chars): {output_text[:200]}")

                # Try multiple version patterns
                version_patterns = [
                    r'(\d{2}\.\d{2}\.\d+\.\d+)',      # Standard: 19.44.35207.0
                    r'Version (\d+\.\d+\.\d+\.\d+)',   # "Version 19.44.35207.0"
                    r'cl\.exe.*?(\d+\.\d+\.\d+\.\d+)',  # cl.exe后面跟着版本号
                    r'(\d{2}\.\d{2}\.\d+)',           # Short: 19.44.35219
                    r'Version (\d+\.\d+\.\d+)',        # "Version 19.44.35219"
                    r'cl\.exe.*?(\d+\.\d+\.\d+)',     # cl.exe后面跟着短版本号
                ]

                for pattern in version_patterns:
                    version_match = re.search(pattern, output_text)
                    if version_match:
                        version = version_match.group(1)
                        print(f"      [DEBUG] Found version with pattern '{pattern}': {version}")

                        # Convert 19.x.x.x to 14.x.x.x format
                        if version.startswith('19.'):
                            if version.count('.') == 2:  # 19.44.35219 -> 14.44.35219
                                version = '14.' + version[3:]
                            else:  # 19.44.35219.x -> 14.44.35219
                                parts = version.split('.')
                                if len(parts) >= 3:
                                    version = f"14.{parts[1]}.{parts[2]}"
                            print(f"      [DEBUG] Converted to MSVC version: {version}")
                        return version

                print(f"      [DEBUG] No version found in output")
            else:
                print(f"      [DEBUG] No output from cl command")
        except Exception as e:
            print(f"      [ERROR] Failed to get MSVC version: {e}")
        return None

    def find_msys2_gcc(self):
        """Find MSYS2 GCC using intelligent search"""
        print(f"  [INFO] Looking for MSYS2 GCC...")

        # 1. Check PATH first
        if shutil.which('gcc'):
            try:
                result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    version_match = re.search(r'gcc.*?(\d+\.\d+\.\d+)', result.stdout)
                    if version_match:
                        version = version_match.group(1)
                        if Requirements.version_compare(version, Requirements.GCC_VERSION) >= 0:
                            gcc_path = shutil.which('gcc')
                            print(f"    {Colors.OK}Found GCC in PATH: {gcc_path} (version {version}){Colors.ENDC}")
                            return {
                                'gcc_path': gcc_path,
                                'version': version,
                                'msys2_path': str(Path(gcc_path).parent.parent.parent),
                                'environment': 'mingw64'
                            }
            except:
                pass

        # 2. Search common MSYS2 locations
        msys2_paths = [
            Path("C:/msys64"),
            Path("D:/msys64"),
            Path("T:/msys64"),
            Path("T:/Softwares/msys64"),
            Path.home() / "msys64",
        ]

        for msys2_path in msys2_paths:
            if msys2_path.exists():
                gcc_path = msys2_path / "mingw64/bin/gcc.exe"
                if gcc_path.exists():
                    try:
                        result = subprocess.run([str(gcc_path), '--version'], capture_output=True, text=True)
                        if result.returncode == 0:
                            version_match = re.search(r'gcc.*?(\d+\.\d+\.\d+)', result.stdout)
                            if version_match:
                                version = version_match.group(1)
                                if Requirements.version_compare(version, Requirements.GCC_VERSION) >= 0:
                                    print(f"    {Colors.OK}Found GCC: {gcc_path} (version {version}){Colors.ENDC}")
                                    return {
                                        'gcc_path': str(gcc_path),
                                        'version': version,
                                        'msys2_path': str(msys2_path),
                                        'environment': 'mingw64'
                                    }
                    except:
                        continue
        return None

    def get_cudnn_lib_path(self, base_path):
        """Get platform-specific cuDNN library path"""
        if self.system == "Windows":
            return base_path / 'lib' / 'x64'
        else:
            # Linux: try lib64 first, then lib
            lib64_path = base_path / 'lib64'
            if lib64_path.exists():
                return lib64_path
            else:
                return base_path / 'lib'

    def check_cuda_cudnn(self):
        """Check CUDA and cuDNN using intelligent search"""
        print(f"\n{Colors.BLUE}[Step 3/7] Checking CUDA and cuDNN...{Colors.ENDC}")

        # Find CUDA
        cuda_paths = self.find_cuda_installations()
        if cuda_paths:
            # Select appropriate version
            cuda_path = self.select_cuda_version(cuda_paths)
            if cuda_path:
                self.config['cuda_root'] = str(cuda_path)
                print(f"  {Colors.GREEN}[OK] CUDA: {cuda_path}{Colors.ENDC}")

                # Find cuDNN (8.x is required)
                cudnn_path = self.find_cudnn_installation()
                if cudnn_path:
                    print(f"  {Colors.GREEN}[OK] cuDNN 8.x: {cudnn_path}{Colors.ENDC}")
                    self.config['cudnn_root'] = str(cudnn_path)
                    # Store bin and lib paths for build scripts
                    self.config['cudnn_bin'] = str(cudnn_path / 'bin')
                    self.config['cudnn_lib'] = str(self.get_cudnn_lib_path(cudnn_path))
                else:
                    # If not found, look in CUDA directory (but still require 8.x)
                    cudnn_found = self.find_cudnn_in_cuda(cuda_path)
                    if cudnn_found and self.is_cudnn_version_8(cuda_path):
                        print(f"  {Colors.GREEN}[OK] cuDNN 8.x: Found in CUDA directory{Colors.ENDC}")
                        self.config['cudnn_root'] = str(cuda_path)
                        self.config['cudnn_bin'] = str(cuda_path / 'bin')
                        self.config['cudnn_lib'] = str(self.get_cudnn_lib_path(cuda_path))
                    else:
                        print(f"  {Colors.FAIL}[ERROR] cuDNN 8.x is required but not found!{Colors.ENDC}")
                        print(f"  {Colors.CYAN}Please install cuDNN 8.x from NVIDIA website{Colors.ENDC}")
                        return False
            else:
                # Prompt user for CUDA path if no compatible version found
                cuda_path = self.prompt_cuda_path()
                if cuda_path:
                    self.config['cuda_root'] = str(cuda_path)
                    print(f"  {Colors.GREEN}[OK] CUDA: {cuda_path}{Colors.ENDC}")

                    # Now try to find cuDNN again with the user-provided CUDA path
                    cudnn_path = self.find_cudnn_installation()
                    if cudnn_path:
                        print(f"  {Colors.GREEN}[OK] cuDNN 8.x: {cudnn_path}{Colors.ENDC}")
                        self.config['cudnn_root'] = str(cudnn_path)
                        self.config['cudnn_bin'] = str(cudnn_path / 'bin')
                        self.config['cudnn_lib'] = str(self.get_cudnn_lib_path(cudnn_path))
                    else:
                        # Prompt for cuDNN path as well
                        cudnn_path = self.prompt_cudnn_path()
                        if cudnn_path:
                            self.config['cudnn_root'] = str(cudnn_path)
                            self.config['cudnn_bin'] = str(cudnn_path / 'bin')
                            self.config['cudnn_lib'] = str(self.get_cudnn_lib_path(cudnn_path))
                            print(f"  {Colors.GREEN}[OK] cuDNN 8.x: {cudnn_path}{Colors.ENDC}")
                        else:
                            print(f"  {Colors.FAIL}[ERROR] cuDNN 8.x is required but not provided!{Colors.ENDC}")
                            return False
                else:
                    print(f"  {Colors.FAIL}[ERROR] CUDA is required but not provided!{Colors.ENDC}")
                    return False
        else:
            # Prompt user for CUDA path if none found
            cuda_path = self.prompt_cuda_path()
            if cuda_path:
                self.config['cuda_root'] = str(cuda_path)
                print(f"  {Colors.GREEN}[OK] CUDA: {cuda_path}{Colors.ENDC}")

                # Try to find cuDNN
                cudnn_path = self.find_cudnn_installation()
                if cudnn_path:
                    print(f"  {Colors.GREEN}[OK] cuDNN 8.x: {cudnn_path}{Colors.ENDC}")
                    self.config['cudnn_root'] = str(cudnn_path)
                    self.config['cudnn_bin'] = str(cudnn_path / 'bin')
                    self.config['cudnn_lib'] = str(self.get_cudnn_lib_path(cudnn_path))
                else:
                    # Prompt for cuDNN path as well
                    cudnn_path = self.prompt_cudnn_path()
                    if cudnn_path:
                        self.config['cudnn_root'] = str(cudnn_path)
                        self.config['cudnn_bin'] = str(cudnn_path / 'bin')
                        self.config['cudnn_lib'] = str(self.get_cudnn_lib_path(cudnn_path))
                        print(f"  {Colors.GREEN}[OK] cuDNN 8.x: {cudnn_path}{Colors.ENDC}")
                    else:
                        print(f"  {Colors.FAIL}[ERROR] cuDNN 8.x is required but not provided!{Colors.ENDC}")
                        return False
            else:
                print(f"  {Colors.FAIL}[ERROR] CUDA is required but not provided!{Colors.ENDC}")
                return False

        # Return True if both CUDA and cuDNN were found successfully
        return 'cuda_root' in self.config and 'cudnn_root' in self.config

    def find_cuda_installations(self):
        """Find all CUDA installations"""
        cuda_paths = []

        if self.system == "Windows":
            search_paths = [
                Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
                Path("C:/Program Files/NVIDIA/CUDA"),
                Path("C:/CUDA")
            ]
        else:
            search_paths = [
                Path("/usr/local/cuda"),  # Linux standard CUDA location
                Path("/opt/cuda"),          # Alternative Linux location
                Path("/usr/cuda"),           # System-wide CUDA
            ]

        for base_path in search_paths:
            if base_path.exists():
                cuda_paths.append(base_path)
                # Also look for versioned directories in /usr/local, /opt, /usr
                if base_path.name != "cuda":  # Avoid duplicate for /usr/local/cuda
                    for cuda_path in base_path.glob("cuda*"):
                        if cuda_path.is_dir():
                            cuda_paths.append(cuda_path)
                    for version_path in base_path.glob("v*"):
                        if version_path.is_dir():
                            # Check if this looks like a CUDA toolkit (contains bin/nvcc)
                            nvcc_path = version_path / "bin" / "nvcc"
                            if nvcc_path.exists():
                                cuda_paths.append(version_path)

        return cuda_paths

    def select_cuda_version(self, cuda_paths):
        """Select compatible CUDA version"""
        required_version = Requirements.CUDA_VERSION

        for cuda_path in cuda_paths:
            # Extract version from path (e.g., "v12.8")
            version_match = re.search(r'v(\d+)\.(\d+)', cuda_path.name)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                path_version = f"{major}.{minor}"

                if Requirements.version_compare(path_version, required_version) >= 0:
                    return cuda_path

        # Return the latest version if no compatible version found
        if cuda_paths:
            return sorted(cuda_paths, reverse=True)[0]

        return None

    def find_cudnn_installation(self):
        """Intelligent cuDNN search - following user's specification"""
        print(f"    [INFO] Searching for cuDNN...")

        # 1. First check PATH environment variable for cuDNN
        path_env = os.environ.get('PATH', '')
        path_dirs = path_env.split(os.pathsep)
        print(f"    [INFO] Checking PATH ({len(path_dirs)} directories)...")

        for path_dir in path_dirs:
            if not path_dir.strip():
                continue
            path_dir = Path(path_dir.strip())
            path_str = str(path_dir).lower()

            # Look for cuDNN in PATH
            if 'cudnn' in path_str:
                print(f"    [INFO] Found cuDNN-related directory in PATH: {path_dir}")
                # If it's a bin directory, infer cuDNN root
                if path_dir.name.lower() == 'bin':
                    cudnn_root = path_dir.parent
                    print(f"    [INFO] Checking cuDNN root from PATH: {cudnn_root}")

                    # Verify the directory structure and version
                    if (cudnn_root / "bin").exists() and (cudnn_root / "lib" / "x64").exists():
                        # Check for cuDNN 8.x version
                        if not self.is_cudnn_version_8(cudnn_root):
                            print(f"    [WARN] Found cuDNN but not version 8.x: {cudnn_root}")
                            continue

                        # Check for key files
                        bin_files = list((cudnn_root / "bin").glob("cudnn*.dll"))
                        cudnn_lib_path = self.get_cudnn_lib_path(cudnn_root)
                        if self.system == "Windows":
                            lib_files = list(cudnn_lib_path.glob("cudnn*.lib"))

                        if bin_files or lib_files:
                            print(f"    {Colors.OK}Found valid cuDNN 8.x in PATH: {cudnn_root}{Colors.ENDC}")
                            print(f"      - DLL files: {len(bin_files)} found")
                            print(f"      - LIB files: {len(lib_files)} found")
                            return cudnn_root
                    else:
                        print(f"    [WARN] Invalid cuDNN directory structure: {cudnn_root}")

        # 2. Check CUDNN_ROOT environment variable
        cudnn_root_env = os.environ.get('CUDNN_ROOT')
        if cudnn_root_env:
            cudnn_root = Path(cudnn_root_env)
            print(f"    [INFO] Checking CUDNN_ROOT: {cudnn_root}")

            if cudnn_root.exists():
                # Check if it's a valid cuDNN directory and version 8.x
                if self.is_valid_cudnn_dir(cudnn_root) and self.is_cudnn_version_8(cudnn_root):
                    print(f"    {Colors.OK}Found valid cuDNN 8.x in CUDNN_ROOT: {cudnn_root}{Colors.ENDC}")
                    return cudnn_root
                elif self.is_valid_cudnn_dir(cudnn_root):
                    print(f"    [WARN] CUDNN_ROOT exists but not version 8.x: {cudnn_root}")
                else:
                    print(f"    [WARN] CUDNN_ROOT exists but has invalid structure: {cudnn_root}")
            else:
                print(f"    [WARN] CUDNN_ROOT doesn't exist: {cudnn_root}")

        # 3. Check common installation locations (only cuDNN 8.x)
        print(f"    [INFO] Searching common cuDNN 8.x installation locations...")
        if self.system == "Windows":
            common_paths = [
                Path("C:/Program Files/NVIDIA/CUDNN"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8.9"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8.9.7"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8.8"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8.7"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8.6"),
                Path("C:/Program Files/NVIDIA/CUDNN/v8.5"),
            ]
        else:
            # Linux cuDNN paths
            common_paths = [
                # CUDA toolkit bundled cuDNN
                Path("/usr/local/cuda"),  # Most common
                Path("/opt/cuda"),          # Alternative
                Path("/usr/cuda"),           # System-wide

                # Standalone cuDNN installations
                Path("/usr/local/cudnn"),
                Path("/opt/cudnn"),
                Path("/usr/local/cuda/include"),  # Check include directory
                Path("/usr/local/cuda/lib64"),      # Check lib64 directory

                # Common locations with version suffixes
                Path("/usr/local/cudnn-8.0"),
                Path("/usr/local/cudnn-8.1"),
                Path("/usr/local/cudnn-8.2"),
                Path("/usr/local/cudnn-8.3"),
                Path("/usr/local/cudnn-8.4"),
                Path("/usr/local/cudnn-8.5"),
                Path("/usr/local/cudnn-8.6"),
                Path("/usr/local/cudnn-8.7"),
                Path("/usr/local/cudnn-8.8"),
                Path("/usr/local/cudnn-8.9"),
            ]

        for base_path in common_paths:
            if base_path.exists():
                print(f"    [INFO] Checking: {base_path}")
                if self.is_valid_cudnn_dir(base_path) and self.is_cudnn_version_8(base_path):
                    print(f"    {Colors.OK}Found valid cuDNN 8.x: {base_path}{Colors.ENDC}")
                    return base_path
                else:
                    # Try version subdirectories (only v8.x)
                    for version_dir in base_path.glob("v8*"):
                        if self.is_valid_cudnn_dir(version_dir):
                            print(f"    {Colors.OK}Found valid cuDNN 8.x version: {version_dir}{Colors.ENDC}")
                            return version_dir

        print(f"    [ERROR] cuDNN 8.x not found in any standard location")
        print(f"    [INFO] cuDNN 8.x is required for this project")
        return None

    def find_cudnn_in_cuda(self, cuda_path):
        """Check if cuDNN is installed in CUDA directory"""
        # Always check for header files (cudnn.h or cudnn_version.h)
        header_files = [
            cuda_path / "include" / "cudnn.h",
            cuda_path / "include" / "cudnn_version.h"  # CUDA 12+ separates version info
        ]

        has_header = any(header.exists() for header in header_files)
        if not has_header:
            return False

        if self.system == "Windows":
            # Windows cuDNN files
            cudnn_files = [
                cuda_path / "bin/cudnn64.dll",
                self.get_cudnn_lib_path(cuda_path) / "cudnn.lib",
            ]
        else:
            # Linux cuDNN files (looking in lib64 for 64-bit systems)
            cudnn_files = [
                cuda_path / "lib64/libcudnn.so",
                cuda_path / "lib64/libcudnn.so.8",  # versioned library
                cuda_path / "lib/libcudnn.so",      # 32-bit fallback
                cuda_path / "lib/libcudnn.so.8",   # 32-bit versioned fallback
            ]

        return any(file.exists() for file in cudnn_files)

    def is_valid_cudnn_dir(self, path):
        """Validate cuDNN directory"""
        # Check for cuDNN header file (always required)
        if not (path / "include" / "cudnn.h").exists():
            return False

        if self.system == "Windows":
            # Windows cuDNN files
            required_files = [
                "bin/cudnn64.dll",           # Windows DLL
            ]
            # Add platform-specific lib path
            lib_path = self.get_cudnn_lib_path(path)
            if lib_path.name == "x64":
                required_files.append("lib/x64/cudnn.lib")
            else:
                required_files.append(f"lib/{lib_path.name}/cudnn.lib")
        else:
            # Linux cuDNN files
            required_files = [
                "lib/libcudnn.so",           # Linux shared library
                "lib64/libcudnn.so",         # Linux 64-bit library (alternative)
                "lib/libcudnn.so.8",         # Linux versioned library
                "lib64/libcudnn.so.8",      # Linux 64-bit versioned library (alternative)
            ]

        # Check if at least one library file exists
        for file_path in required_files:
            if (path / file_path).exists():
                return True

        return False

    def is_cudnn_version_8(self, path):
        """Check if cuDNN version is 8.x"""
        # Method 1: Check directory name for version
        dir_name = path.name.lower()
        if dir_name.startswith('v8') or '8.' in dir_name:
            return True

        # Method 2: Check parent directory name if this is a subdirectory
        parent_name = path.parent.name.lower()
        if parent_name.startswith('v8') or '8.' in parent_name:
            return True

        # Method 3: Check cuDNN header files for version
        header_files = [
            path / "include/cudnn.h",
            path / "include/cudnn_version.h"  # CUDA 12+ separates version info
        ]

        for header_file in header_files:
            if header_file.exists():
                try:
                    with open(header_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Look for CUDNN_MAJOR define (newer format)
                        if '#define CUDNN_MAJOR 8' in content:
                            return True
                        # Look for CUDNN_MAJOR_VERSION define (older format)
                        if '#define CUDNN_MAJOR_VERSION 8' in content:
                            return True
                        # Also check for version in comments
                        if 'cudnn 8.' in content.lower():
                            return True
                except:
                    pass

        # Method 4: Check DLL file version info (Windows)
        dll_files = list((path / "bin").glob("cudnn*.dll"))
        if dll_files:
            # Check DLL file names for version
            for dll_file in dll_files:
                dll_name = dll_file.name.lower()
                if 'cudnn64_8' in dll_name or 'cudnn8' in dll_name:
                    return True

        return False

    def check_python(self) -> bool:
        """Check Python and NumPy using intelligent search"""
        print(f"\n{Colors.BLUE}[Step 4/7] Checking Python and NumPy...{Colors.ENDC}")

        # 1. Try to get Python from system PATH
        python_path = shutil.which('python')
        if python_path:
            if self.validate_python(python_path):
                return True

        # 2. Try 'python3' from system PATH
        python3_path = shutil.which('python3')
        if python3_path:
            if self.validate_python(python3_path):
                return True

        # 3. Check common Python installation locations
        print(f"  {Colors.INFO}Searching for Python installations...{Colors.ENDC}")
        if self.system == "Windows":
            python_paths = [
                Path.home() / "AppData/Local/Programs/Python",
                Path("C:/Python312"),
                Path("C:/Python311"),
                Path("C:/Python310"),
                Path("C:/Python39"),
                Path("C:/Python38"),
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Python",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Python",
                Path("C:/Python"),
            ]

            for python_dir in python_paths:
                if python_dir.exists():
                    for version_dir in python_dir.glob("Python*"):
                        python_exe = version_dir / "python.exe"
                        if python_exe.exists() and self.validate_python(str(python_exe)):
                            return True
        else:
            # Linux/macOS common locations
            python_paths = [
                Path("/usr/bin/python3"),
                Path("/usr/local/bin/python3"),
                Path("/opt/python3/bin/python3"),
                Path.home() / ".local/bin/python3",
            ]

            for python_exe in python_paths:
                if python_exe.exists() and self.validate_python(str(python_exe)):
                    return True

        # 4. Prompt user for Python path
        return self.prompt_python_path()

    def prompt_python_confirmation(self, python_path, version, numpy_version):
        """Prompt user to confirm using the detected Python installation"""
        print(f"  {Colors.CYAN}Use this Python installation?{Colors.ENDC}")
        print(f"    Path: {python_path}")
        print(f"    Version: Python {version}")
        if numpy_version:
            print(f"    NumPy: {numpy_version}")
        else:
            print(f"    NumPy: Not found or incompatible")

        while True:
            choice = input(f"  {Colors.CYAN}Use this Python? (y/n): {Colors.ENDC}").strip().lower()
            if choice == 'y' or choice == 'yes':
                # User confirmed, save configuration
                self.config['python'] = python_path
                self.config['python_version'] = version
                if numpy_version:
                    self.config['numpy_version'] = numpy_version
                else:
                    self.config['numpy_version'] = Requirements.NUMPY_VERSION
                    print(f"  {Colors.INFO}NumPy not found, using required version: {Requirements.NUMPY_VERSION}{Colors.ENDC}")
                return True
            elif choice == 'n' or choice == 'no':
                # User rejected, prompt for custom path
                return self.prompt_custom_python_path()
            else:
                print(f"  {Colors.WARN}Please enter 'y' for yes or 'n' for no{Colors.ENDC}")

    def prompt_custom_python_path(self):
        """Prompt user to provide custom Python path"""
        print(f"  {Colors.INFO}Please provide the path to your Python installation{Colors.ENDC}")
        print(f"  {Colors.INFO}Common locations:{Colors.ENDC}")

        if self.system == "Windows":
            print(f"    - C:\\Python314\\python.exe")
            print(f"    - C:\\Python313\\python.exe")
            print(f"    - <venv-path>\\Scripts\\python.exe")
            print(f"    - <conda-env-path>\\python.exe")
        else:
            print(f"    - /usr/bin/python3")
            print(f"    - /usr/local/bin/python3")
            print(f"    - <venv-path>/bin/python3")
            print(f"    - <conda-env-path>/bin/python3")

        while True:
            path_input = input(f"  {Colors.CYAN}Enter Python path (or press Enter to skip): {Colors.ENDC}").strip().replace('"', '')
            if not path_input:
                print(f"  {Colors.WARN}Skipping Python configuration{Colors.ENDC}")
                return False

            python_path = Path(path_input)

            # Adjust for Windows
            if self.system == "Windows" and python_path.suffix != '.exe':
                python_path = python_path / 'python.exe'

            if python_path.exists():
                # Validate the custom Python path
                try:
                    result = subprocess.run([str(python_path), '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        version_str = result.stderr.strip() if result.stderr else result.stdout.strip()
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', version_str)
                        if version_match:
                            version = version_match.group(1)
                            if Requirements.version_compare(version, Requirements.PYTHON_MIN) >= 0:
                                # Check NumPy for custom path
                                numpy_version = None
                                try:
                                    numpy_result = subprocess.run([str(python_path), '-c', 'import numpy; print(numpy.__version__)'],
                                                                  capture_output=True, text=True, timeout=10)
                                    if numpy_result.returncode == 0:
                                        numpy_version = numpy_result.stdout.strip()
                                except:
                                    numpy_version = None

                                print(f"  {Colors.GREEN}[OK] Using Python {version}: {python_path}{Colors.ENDC}")
                                if numpy_version:
                                    print(f"  {Colors.GREEN}[OK] NumPy {numpy_version}{Colors.ENDC}")
                                    self.config['numpy_version'] = numpy_version
                                else:
                                    print(f"  {Colors.INFO}NumPy not found, using required version: {Requirements.NUMPY_VERSION}{Colors.ENDC}")
                                    self.config['numpy_version'] = Requirements.NUMPY_VERSION

                                self.config['python'] = str(python_path)
                                self.config['python_version'] = version
                                return True
                            else:
                                print(f"  {Colors.WARN}[ERROR] Python version too low: {version} < {Requirements.PYTHON_MIN}{Colors.ENDC}")
                        else:
                            print(f"  {Colors.WARN}[ERROR] Unable to parse Python version{Colors.ENDC}")
                    else:
                        print(f"  {Colors.WARN}[ERROR] Not a valid Python executable: {python_path}{Colors.ENDC}")
                except Exception as e:
                    print(f"  {Colors.WARN}[ERROR] Failed to validate Python: {e}{Colors.ENDC}")
            else:
                print(f"  {Colors.WARN}[ERROR] File not found: {python_path}{Colors.ENDC}")

            retry = input(f"  {Colors.CYAN}Try again? (y/n): {Colors.ENDC}").strip().lower()
            if retry != 'y':
                print(f"  {Colors.WARN}Skipping Python configuration{Colors.ENDC}")
                return False

    def validate_python(self, python_path):
        """Validate Python installation and check version"""
        try:
            # Check Python version
            result = subprocess.run([python_path, '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                return False

            # Parse version string
            version_str = result.stderr.strip() if result.stderr else result.stdout.strip()
            version_match = re.search(r'Python (\d+\.\d+\.\d+)', version_str)
            if version_match:
                version = version_match.group(1)
                if Requirements.version_compare(version, Requirements.PYTHON_MIN) >= 0:
                    # Check NumPy first before asking for confirmation
                    numpy_version = None
                    try:
                        numpy_result = subprocess.run([python_path, '-c', 'import numpy; print(numpy.__version__)'],
                                                      capture_output=True, text=True, timeout=10)
                        if numpy_result.returncode == 0:
                            numpy_version = numpy_result.stdout.strip()
                        else:
                            numpy_version = None
                    except:
                        numpy_version = None

                    # Show detected Python information
                    print(f"  {Colors.GREEN}[OK] Found Python {version}: {python_path}{Colors.ENDC}")
                    if numpy_version:
                        print(f"  {Colors.GREEN}[OK] NumPy {numpy_version}{Colors.ENDC}")

                    # Ask user if they want to use this Python installation
                    return self.prompt_python_confirmation(python_path, version, numpy_version)
                else:
                    print(f"  {Colors.WARN}[ERROR] Python version too low: {version} < {Requirements.PYTHON_MIN}{Colors.ENDC}")
            else:
                print(f"  {Colors.WARN}[ERROR] Unable to parse Python version{Colors.ENDC}")
        except Exception as e:
            print(f"  {Colors.WARN}[ERROR] Failed to validate Python: {e}{Colors.ENDC}")
        return False

    def prompt_python_path(self):
        """Prompt user for Python path"""
        print(f"  {Colors.WARN}[ERROR] Python not found{Colors.ENDC}")

        while True:
            path_input = input(f"  {Colors.CYAN}Enter full path to python.exe (or press Enter to skip): {Colors.ENDC}").strip().replace('"', '')
            if not path_input:
                print(f"  {Colors.WARN}Skipping Python configuration{Colors.ENDC}")
                return False

            python_path = Path(path_input)
            if python_path.name != 'python.exe':
                python_path = python_path / 'python.exe'

            if python_path.exists() and self.validate_python(str(python_path)):
                return True
            else:
                print(f"  {Colors.FAIL}[ERROR] Invalid Python executable: {python_path}{Colors.ENDC}")

    def check_other_dependencies(self):
        """Check other dependencies like Eigen3 and SIMD"""
        print(f"\n{Colors.BLUE}[Step 5/7] Checking other dependencies...{Colors.ENDC}")

        # Check Eigen3 through vcpkg
        eigen3_found = self.check_eigen3()
        if not eigen3_found:
            self.suggest_eigen3_installation()

        # Check SIMD library through vcpkg
        simd_found = self.check_simd()
        if not simd_found:
            self.suggest_simd_installation()

    def check_eigen3(self) -> bool:
        """Check if Eigen is installed"""
        print(f"  [INFO] Checking Eigen...")

        # First, try to find Eigen automatically
        if self._try_find_eigen_auto():
            return True

        # If not found automatically, ask user for manual path
        print(f"  {Colors.CYAN}Eigen not found automatically{Colors.ENDC}")
        print(f"  {Colors.INFO}Please provide the path to Eigen installation{Colors.ENDC}")
        print(f"  {Colors.INFO}Common locations:{Colors.ENDC}")

        if self.system == "Windows":
            print(f"    - .\\third_party\\Eigen")
            print(f"    - .\\vendor\\Eigen")
            print(f"    - C:\\Eigen")
            print(f"    - C:\\libs\\Eigen")
            print(f"    - <vcpkg-root>\\installed\\x64-windows\\include\\Eigen")
        else:
            print(f"    - ./third_party/Eigen")
            print(f"    - ./vendor/Eigen")
            print(f"    - /usr/local/include/Eigen")
            print(f"    - /opt/Eigen")
            print(f"    - <vcpkg-root>/installed/x64-linux/include/Eigen")

        while True:
            eigen_path_input = input(f"  {Colors.CYAN}Enter Eigen path (or press Enter to skip): {Colors.ENDC}").strip()

            if not eigen_path_input:
                # User skipped, will show download suggestions
                return False

            eigen_path = Path(eigen_path_input)

            if self._validate_eigen_path(eigen_path):
                return True
            else:
                print(f"  {Colors.WARN}[ERROR] Invalid Eigen path or Eigen not found at: {eigen_path}{Colors.ENDC}")
                retry = input(f"  {Colors.CYAN}Try again? (y/n): {Colors.ENDC}").strip().lower()
                if retry != 'y':
                    return False

    def _try_find_eigen_auto(self) -> bool:
        """Try to find Eigen3 automatically"""
        # Method 1: Check through vcpkg
        if self._find_eigen_in_vcpkg():
            return True

        # Method 2: Check system PATH environment variable
        if self._find_eigen_in_path():
            return True

        # Method 3: Check common system locations
        if self._find_eigen_in_system_paths():
            return True

        return False

    def _find_eigen_in_vcpkg(self) -> bool:
        """Find Eigen3 through vcpkg"""
        vcpkg_root = self.config.get('vcpkg_root')
        if not vcpkg_root:
            return False

        vcpkg_installed = Path(vcpkg_root) / "installed"
        if not vcpkg_installed.exists():
            return False

        # Check platform-specific vcpkg directories
        if self.system == "Windows":
            vcpkg_dirs = ["x64-windows"]
        else:
            vcpkg_dirs = ["x64-linux", "x64-osx"]

        for vcpkg_dir in vcpkg_dirs:
            base_path = vcpkg_installed / vcpkg_dir
            # Prioritize header file paths over config file paths
            eigen_locations = [
                base_path / "include" / "eigen3" / "Eigen",    # Standard vcpkg eigen3 installation
                base_path / "include" / "Eigen",              # Alternative eigen installation
                base_path / "include" / "eigen3",             # eigen3 directory (for include path)
                base_path / "share" / "eigen3",               # CMake config files
                base_path / "share" / "eigen3-config.cmake"    # CMake config file
            ]

            for location in eigen_locations:
                if location.exists():
                    return self._save_eigen_location(location)

        return False

    def _find_eigen_in_path(self) -> bool:
        """Find Eigen by searching system PATH environment variable"""
        print(f"    {Colors.INFO}Searching system PATH for Eigen...{Colors.ENDC}")

        # Get current project directory and check for third_party first
        current_dir = Path(__file__).parent
        project_third_party_paths = [
            current_dir / "third_party" / "Eigen",
            current_dir / "third_party" / "eigen3",
            current_dir / "vendor" / "Eigen",
            current_dir / "vendor" / "eigen3",
            current_dir / "external" / "Eigen",
            current_dir / "external" / "eigen3",
            current_dir / "deps" / "Eigen",
            current_dir / "deps" / "eigen3",
        ]

        # Check project-specific paths first (highest priority)
        for check_path in project_third_party_paths:
            if check_path.exists() and self._is_valid_eigen_location(check_path):
                print(f"    {Colors.OK}Found Eigen in project: {check_path}{Colors.ENDC}")
                return self._save_eigen_location(check_path)

        # Get system PATH
        path_env = os.environ.get('PATH', '')
        path_dirs = path_env.split(os.pathsep)

        for path_dir in path_dirs:
            if not path_dir.strip():
                continue

            try:
                path_path = Path(path_dir.strip())

                # Check if this directory contains Eigen
                eigen_check_paths = [
                    path_path / "Eigen",             # Eigen directly in PATH
                    path_path / "include" / "Eigen",  # include/Eigen
                    path_path.parent / "include" / "Eigen",  # sibling include dir
                    path_path.parent / "Eigen",          # sibling Eigen dir
                ]

                for check_path in eigen_check_paths:
                    if check_path.exists() and self._is_valid_eigen_location(check_path):
                        print(f"    {Colors.OK}Found Eigen in PATH: {check_path}{Colors.ENDC}")
                        return self._save_eigen_location(check_path)

                # Check if we're in a directory that might contain Eigen
                if path_path.name.lower() in ['eigen', 'eigen-library']:
                    if self._is_valid_eigen_location(path_path):
                        print(f"    {Colors.OK}Found Eigen in PATH: {path_path}{Colors.ENDC}")
                        return self._save_eigen_location(path_path)

            except (OSError, PermissionError):
                # Skip invalid paths
                continue

        return False

    def _find_eigen_in_system_paths(self) -> bool:
        """Find Eigen in common system installation locations"""
        print(f"    {Colors.INFO}Searching common system locations for Eigen...{Colors.ENDC}")

        system_paths = []

        # Get current project directory (where configure.py is located)
        current_dir = Path(__file__).parent

        # Add project-specific paths first
        project_paths = [
            current_dir / "third_party" / "Eigen",
            current_dir / "third_party" / "eigen3",
            current_dir / "vendor" / "Eigen",
            current_dir / "vendor" / "eigen3",
            current_dir / "external" / "Eigen",
            current_dir / "external" / "eigen3",
            current_dir / "deps" / "Eigen",
            current_dir / "deps" / "eigen3",
            current_dir / "Eigen",
            current_dir / "eigen3",
        ]
        system_paths.extend(project_paths)

        # Add system-wide paths
        if self.system == "Linux":
            system_paths.extend([
                Path("/usr/local/include/Eigen"),
                Path("/usr/include/Eigen"),
                Path("/opt/Eigen"),
                Path.home() / ".local" / "include" / "Eigen",
                Path.home() / "Eigen",
            ])
        elif self.system == "Windows":
            system_paths.extend([
                Path("C:/Eigen"),
                Path("C:/libs/Eigen"),
                Path.home() / "Eigen",
                Path.home() / "libs" / "Eigen",
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Eigen",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Eigen",
            ])

        for location in system_paths:
            if location.exists() and self._is_valid_eigen_location(location):
                return self._save_eigen_location(location)

        return False

    def _is_valid_eigen_location(self, location: Path) -> bool:
        """Check if a location contains a valid Eigen installation"""
        # Check for Eigen signature files
        signature_files = [
            "Eigen/Core",
            "Core",  # direct Eigen directory
            "signature_of_eigen3_matrix_library",  # Eigen signature file
        ]

        for sig_file in signature_files:
            if (location / sig_file).exists():
                return True

        # Also check for key Eigen headers
        eigen_headers = ["Core", "Geometry", "Dense", "Sparse"]
        found_headers = 0
        for header in eigen_headers:
            if (location / header).exists() or (location / "Eigen" / header).exists():
                found_headers += 1

        return found_headers >= 2  # At least 2 key headers should exist

    def _validate_eigen_path(self, eigen_path: Path) -> bool:
        """Validate user-provided Eigen path"""
        if not eigen_path.exists():
            return False

        # Check for Eigen signature files
        eigen_signature_files = [
            "Eigen/Core",
            "Core",  # direct Eigen directory
            "signature_of_eigen3_matrix_library",  # Eigen signature file
        ]

        for sig_file in eigen_signature_files:
            if (eigen_path / sig_file).exists():
                return self._save_eigen_location(eigen_path)

        # If user pointed to the parent of Eigen directory
        for subdir in ["Eigen"]:
            subdir_path = eigen_path / subdir
            if subdir_path.exists() and self._is_valid_eigen_location(subdir_path):
                return self._save_eigen_location(subdir_path)

        return False

    def _save_eigen_location(self, location: Path) -> bool:
        """Save the found Eigen location"""
        if location.name in ["Eigen", "eigen3"]:
            print(f"  {Colors.GREEN}[OK] Eigen found: {location}{Colors.ENDC}")
            # Save Eigen include path (parent of Eigen directory)
            self.config['eigen3_include'] = str(location.parent)
            return True
        elif location.name.endswith("-config.cmake"):
            print(f"  {Colors.GREEN}[OK] Eigen found via CMake config: {location}{Colors.ENDC}")
            # Save Eigen config path
            self.config['eigen3_config'] = str(location)
            return True
        else:
            print(f"  {Colors.GREEN}[OK] Eigen found: {location}{Colors.ENDC}")
            # Save Eigen share directory
            self.config['eigen3_share'] = str(location)
            return True

    def suggest_eigen3_installation(self):
        """Suggest Eigen3 installation via vcpkg"""
        print(f"\n{Colors.CYAN}Eigen installation suggestions:{Colors.ENDC}")
        vcpkg_root = self.config.get('vcpkg_root')

        if vcpkg_root:
            print(f"  Install Eigen3 with vcpkg:")
            print(f"    cd \"{vcpkg_root}\"")
            if self.system == "Windows":
                print(f"    .\\vcpkg install eigen3")
            else:
                print(f"    ./vcpkg install eigen3")
            print(f"    Note: Eigen3 includes OpenMP support by default")
        else:
            print(f"  vcpkg is not configured. Install vcpkg first, then:")
            if self.system == "Windows":
                print(f"    .\\vcpkg install eigen3")
            else:
                print(f"    ./vcpkg install eigen3")

    def check_simd(self) -> bool:
        """Check if SIMD library is installed"""
        print(f"  [INFO] Checking SIMD library...")

        # First, try to find SIMD automatically
        if self._try_find_simd_auto():
            return True

        # If not found automatically, ask user for manual path
        print(f"  {Colors.CYAN}SIMD library not found automatically{Colors.ENDC}")
        print(f"  {Colors.INFO}Please provide the path to SIMD library installation{Colors.ENDC}")
        print(f"  {Colors.INFO}Common locations:{Colors.ENDC}")

        if self.system == "Windows":
            print(f"    - .\\third_party\\Simd")
            print(f"    - .\\vendor\\Simd")
            print(f"    - C:\\Simd")
            print(f"    - C:\\libs\\Simd")
            print(f"    - <vcpkg-root>\\installed\\x64-windows\\include\\Simd")
        else:
            print(f"    - ./third_party/Simd")
            print(f"    - ./vendor/Simd")
            print(f"    - /usr/local/include/Simd")
            print(f"    - /opt/Simd")
            print(f"    - <vcpkg-root>/installed/x64-linux/include/Simd")

        while True:
            simd_path_input = input(f"  {Colors.CYAN}Enter SIMD path (or press Enter to skip): {Colors.ENDC}").strip()

            if not simd_path_input:
                # User skipped, will show download suggestions
                return False

            simd_path = Path(simd_path_input)

            if self._validate_simd_path(simd_path):
                return True
            else:
                print(f"  {Colors.WARN}[ERROR] Invalid SIMD path or SIMD library not found at: {simd_path}{Colors.ENDC}")
                retry = input(f"  {Colors.CYAN}Try again? (y/n): {Colors.ENDC}").strip().lower()
                if retry != 'y':
                    return False

    def _try_find_simd_auto(self) -> bool:
        """Try to find SIMD library automatically"""
        # Method 1: Check through vcpkg
        if self._find_simd_in_vcpkg():
            return True

        # Method 2: Check system PATH environment variable
        if self._find_simd_in_path():
            return True

        # Method 3: Check common system locations
        if self._find_simd_in_system_paths():
            return True

        return False

    def _find_simd_in_vcpkg(self) -> bool:
        """Find SIMD library through vcpkg"""
        vcpkg_root = self.config.get('vcpkg_root')
        if not vcpkg_root:
            return False

        vcpkg_installed = Path(vcpkg_root) / "installed"
        if not vcpkg_installed.exists():
            return False

        # Check platform-specific vcpkg directories
        if self.system == "Windows":
            vcpkg_dirs = ["x64-windows"]
        else:
            vcpkg_dirs = ["x64-linux", "x64-osx"]

        for vcpkg_dir in vcpkg_dirs:
            base_path = vcpkg_installed / vcpkg_dir
            simd_locations = [
                base_path / "include" / "Simd",
                base_path / "share" / "simd",
                base_path / "share" / "simd-config.cmake"
            ]

            for location in simd_locations:
                if location.exists():
                    return self._save_simd_location(location)

        return False

    def _find_simd_in_path(self) -> bool:
        """Find SIMD library by searching system PATH environment variable"""
        print(f"    {Colors.INFO}Searching system PATH for SIMD library...{Colors.ENDC}")

        # Get current project directory and check for third_party first
        current_dir = Path(__file__).parent
        project_third_party_paths = [
            current_dir / "third_party" / "Simd",
            current_dir / "third_party" / "simd",
            current_dir / "vendor" / "Simd",
            current_dir / "vendor" / "simd",
            current_dir / "external" / "Simd",
            current_dir / "external" / "simd",
            current_dir / "deps" / "Simd",
            current_dir / "deps" / "simd",
        ]

        # Check project-specific paths first (highest priority)
        for check_path in project_third_party_paths:
            if check_path.exists() and self._is_valid_simd_location(check_path):
                print(f"    {Colors.OK}Found SIMD library in project: {check_path}{Colors.ENDC}")
                return self._save_simd_location(check_path)

        # Get system PATH
        path_env = os.environ.get('PATH', '')
        path_dirs = path_env.split(os.pathsep)

        for path_dir in path_dirs:
            if not path_dir.strip():
                continue

            try:
                path_path = Path(path_dir.strip())

                # Check if this directory contains SIMD
                if path_path.name.lower() in ['simd', 'simd-library']:
                    if self._is_valid_simd_location(path_path):
                        print(f"    {Colors.OK}Found SIMD library in PATH: {path_path}{Colors.ENDC}")
                        return self._save_simd_location(path_path)

                # Check for include subdirectories
                include_subdirs = ['include', 'includes', 'inc']
                for subdir in include_subdirs:
                    simd_subdir = path_path / subdir / "Simd"
                    if simd_subdir.exists() and self._is_valid_simd_location(simd_subdir):
                        print(f"    {Colors.OK}Found SIMD library in PATH: {simd_subdir}{Colors.ENDC}")
                        return self._save_simd_location(simd_subdir)

            except (OSError, PermissionError):
                # Skip invalid paths
                continue

        return False

    def _find_simd_in_system_paths(self) -> bool:
        """Find SIMD library in common system installation locations"""
        print(f"    {Colors.INFO}Searching common system locations for SIMD library...{Colors.ENDC}")

        system_paths = []

        # Get current project directory (where configure.py is located)
        current_dir = Path(__file__).parent

        # Add project-specific paths first
        project_paths = [
            current_dir / "third_party" / "Simd",
            current_dir / "third_party" / "simd",
            current_dir / "vendor" / "Simd",
            current_dir / "vendor" / "simd",
            current_dir / "external" / "Simd",
            current_dir / "external" / "simd",
            current_dir / "deps" / "Simd",
            current_dir / "deps" / "simd",
            current_dir / "Simd",
            current_dir / "simd",
        ]
        system_paths.extend(project_paths)

        # Add system-wide paths
        if self.system == "Linux":
            system_paths.extend([
                Path("/usr/local/include/Simd"),
                Path("/usr/include/Simd"),
                Path("/opt/Simd"),
                Path.home() / ".local" / "include" / "Simd",
                Path.home() / "Simd",
            ])
        elif self.system == "Windows":
            system_paths.extend([
                Path("C:/Simd"),
                Path("C:/libs/Simd"),
                Path.home() / "Simd",
                Path.home() / "libs" / "Simd",
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Simd",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Simd",
            ])

        for location in system_paths:
            if location.exists() and self._is_valid_simd_location(location):
                return self._save_simd_location(location)

        return False

    def _is_valid_simd_location(self, location: Path) -> bool:
        """Check if a location contains a valid SIMD library installation"""
        # Check for SIMD signature files
        signature_files = [
            "Simd/Simd.h",
            "Simd.h",  # direct Simd directory
            "SimdLib.h",  # alternative header name
        ]

        for sig_file in signature_files:
            if (location / sig_file).exists():
                return True

        # Also check for key SIMD headers
        simd_headers = ["Simd.h", "SimdLib.h", "SimdParallel.h"]
        found_headers = 0
        for header in simd_headers:
            if (location / header).exists() or (location / "Simd" / header).exists():
                found_headers += 1

        return found_headers >= 1  # At least 1 key header should exist

    def _validate_simd_path(self, simd_path: Path) -> bool:
        """Validate user-provided SIMD library path"""
        if not simd_path.exists():
            return False

        # Check for SIMD signature files
        simd_signature_files = [
            "Simd/Simd.h",
            "Simd.h",  # direct Simd directory
            "SimdLib.h",  # alternative header name
        ]

        for sig_file in simd_signature_files:
            if (simd_path / sig_file).exists():
                return self._save_simd_location(simd_path)

        return False

    def _save_simd_location(self, location: Path) -> bool:
        """Save the found SIMD library location"""
        if location.name in ["Simd", "simd"]:
            print(f"  {Colors.GREEN}[OK] SIMD library found: {location}{Colors.ENDC}")
            # Save SIMD include path (parent of Simd directory)
            self.config['simd_include'] = str(location.parent)
            return True
        elif location.name.endswith("-config.cmake"):
            print(f"  {Colors.GREEN}[OK] SIMD library found via CMake config: {location}{Colors.ENDC}")
            # Save SIMD config path
            self.config['simd_config'] = str(location)
            return True
        else:
            print(f"  {Colors.GREEN}[OK] SIMD library found: {location}{Colors.ENDC}")
            # Save SIMD share directory
            self.config['simd_share'] = str(location)
            return True

    def suggest_simd_installation(self):
        """Suggest SIMD library installation via vcpkg"""
        print(f"\n{Colors.CYAN}SIMD library installation suggestions:{Colors.ENDC}")
        vcpkg_root = self.config.get('vcpkg_root')

        if vcpkg_root:
            print(f"  Install SIMD library with vcpkg:")
            print(f"    cd \"{vcpkg_root}\"")
            if self.system == "Windows":
                print(f"    .\\vcpkg install simd")
            else:
                print(f"    ./vcpkg install simd")
            print(f"    Note: SIMD library provides optimized image processing functions")
        else:
            print(f"  vcpkg is not configured. Install vcpkg first, then:")
            if self.system == "Windows":
                print(f"    .\\vcpkg install simd")
            else:
                print(f"    ./vcpkg install simd")

    def prompt_cuda_path(self):
        """Prompt user for CUDA path"""
        print(f"  {Colors.WARN}[ERROR] CUDA not found{Colors.ENDC}")
        print(f"  {Colors.CYAN}CUDA is required for this project{Colors.ENDC}")

        while True:
            path_input = input(f"  {Colors.CYAN}Enter CUDA installation path (e.g., C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8): {Colors.ENDC}").strip().replace('"', '')
            if not path_input:
                print(f"  {Colors.WARN}Skipping CUDA configuration{Colors.ENDC}")
                return None

            cuda_path = Path(path_input)
            if cuda_path.exists() and (cuda_path / "bin").exists():
                return cuda_path
            else:
                print(f"  {Colors.FAIL}[ERROR] Invalid CUDA installation: {cuda_path}{Colors.ENDC}")
                print(f"  {Colors.INFO}CUDA directory should contain 'bin' subdirectory{Colors.ENDC}")

    def prompt_cudnn_path(self):
        """Prompt user for cuDNN 8.x path"""
        print(f"  {Colors.WARN}[ERROR] cuDNN 8.x not found{Colors.ENDC}")
        print(f"  {Colors.CYAN}cuDNN 8.x is required for this project{Colors.ENDC}")

        while True:
            path_input = input(f"  {Colors.CYAN}Enter cuDNN 8.x installation path (e.g., C:/Program Files/NVIDIA/CUDNN/v8.9.7): {Colors.ENDC}").strip().replace('"', '')
            if not path_input:
                print(f"  {Colors.WARN}Skipping cuDNN configuration{Colors.ENDC}")
                return None

            cudnn_path = Path(path_input)
            if cudnn_path.exists():
                # Verify it's a valid cuDNN 8.x installation
                if self.is_valid_cudnn_dir(cudnn_path) and self.is_cudnn_version_8(cudnn_path):
                    return cudnn_path
                elif not self.is_valid_cudnn_dir(cudnn_path):
                    print(f"  {Colors.FAIL}[ERROR] Invalid cuDNN directory structure: {cudnn_path}{Colors.ENDC}")
                    lib_path = self.get_cudnn_lib_path(cudnn_path)
                    relative_lib_path = lib_path.relative_to(cudnn_path)
                    print(f"  {Colors.INFO}cuDNN directory should contain include/, bin/, and {relative_lib_path}/ subdirectories{Colors.ENDC}")
                elif not self.is_cudnn_version_8(cudnn_path):
                    print(f"  {Colors.FAIL}[ERROR] cuDNN version is not 8.x: {cudnn_path}{Colors.ENDC}")
                    print(f"  {Colors.INFO}This project requires cuDNN 8.x specifically{Colors.ENDC}")
            else:
                print(f"  {Colors.FAIL}[ERROR] Directory does not exist: {cudnn_path}{Colors.ENDC}")

    def show_summary(self):
        """Show configuration summary"""
        print(f"\n{Colors.BLUE}[Step 6/7] Configuration Summary...{Colors.ENDC}")
        print(f"  {Colors.GREEN}Configuration completed successfully!{Colors.ENDC}")

    def run(self):
        """Main configuration run"""
        print(f"{Colors.HEADER}{Colors.BOLD}=== Smart Project Configuration Wizard ==={Colors.ENDC}")
        print(f"{Colors.INFO}System: {self.system}{Colors.ENDC}")

        # 1. Check basic build tools (intelligent detection)
        if not self.check_basic_tools():
            return False

        # 2. Setup compiler using hardcoded paths for now
        if not self.setup_compiler():
            return False

        # 3. Check CUDA and cuDNN (required for this project)
        if not self.check_cuda_cudnn():
            return False

        # 4. Check Python and NumPy
        if not self.check_python():
            return False

        # 5. Check other dependencies
        self.check_other_dependencies()

        # 6. Generate configurations
        self.generate_configurations()

        # 7. Show summary and next steps
        self.show_summary()
        return True

    def generate_configurations(self):
        """Generate configuration files"""
        print(f"\n{Colors.BLUE}Generating configuration files...{Colors.ENDC}")

        # Create config directory
        self.cmake_cache.parent.mkdir(exist_ok=True)

        # 1. Generate JSON configuration
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"  {Colors.GREEN}[OK] Generated {self.config_file}{Colors.ENDC}")

        # 2. Generate CMake configuration
        self.generate_cmake_config()

        # 3. Generate build scripts
        self.generate_build_scripts()

    def generate_cmake_config(self):
        """Generate CMake configuration file"""
        with open(self.cmake_cache, 'w') as f:
            f.write("# Auto-generated configuration file\n")
            f.write("# Generated by configure.py\n\n")

            # Only include configurations for tools that were found
            if 'cuda_root' in self.config:
                cuda_path = self.config["cuda_root"].replace('\\', '/')
                f.write(f'set(CUDAToolkit_ROOT "{cuda_path}")\n')
            if 'cudnn_root' in self.config:
                cudnn_root = self.config["cudnn_root"].replace('\\', '/')
                f.write(f'set(CUDNN_ROOT "{cudnn_root}")\n')
            if 'cudnn_bin' in self.config:
                cudnn_bin = self.config["cudnn_bin"].replace('\\', '/')
                f.write(f'set(CUDNN_BIN "{cudnn_bin}")\n')
            if 'cudnn_lib' in self.config:
                cudnn_lib = self.config["cudnn_lib"].replace('\\', '/')
                f.write(f'set(CUDNN_LIB "{cudnn_lib}")\n')
            if 'python' in self.config:
                python_path = self.config["python"].replace('\\', '/')
                f.write(f'set(Python3_EXECUTABLE "{python_path}")\n')
            if 'vcpkg_root' in self.config:
                vcpkg_root = self.config["vcpkg_root"].replace('\\', '/')
                f.write(f'set(CMAKE_TOOLCHAIN_FILE "{vcpkg_root}/scripts/buildsystems/vcpkg.cmake")\n')

            # MSVC configuration
            msvc_info = self.config.get('compilers', {}).get('msvc', {})
            if msvc_info:
                f.write(f'\n# MSVC configuration\n')
                if 'cl_path' in msvc_info:
                    cl_path = msvc_info["cl_path"].replace('\\', '/')
                    f.write(f'set(cl_path "{cl_path}")\n')
                if 'vcvars_path' in msvc_info:
                    vcvars_path = msvc_info["vcvars_path"].replace('\\', '/')
                    f.write(f'set(vcvars_path "{vcvars_path}")\n')
                if 'vs_path' in msvc_info:
                    vs_path = msvc_info["vs_path"].replace('\\', '/')
                    f.write(f'set(vs_path "{vs_path}")\n')

            # MSYS2 configuration
            msys2_info = self.config.get('compilers', {}).get('msys2', {})
            if msys2_info:
                f.write(f'\n# MSYS2 configuration\n')
                if 'gcc_path' in msys2_info:
                    gcc_path = msys2_info["gcc_path"].replace('\\', '/')
                    f.write(f'set(gcc_path "{gcc_path}")\n')
                if 'msys2_path' in msys2_info:
                    msys2_path = msys2_info["msys2_path"].replace('\\', '/')
                    f.write(f'set(msys2_path "{msys2_path}")\n')
                if 'environment' in msys2_info:
                    f.write(f'set(environment "{msys2_info["environment"]}")\n')

            # Ninja path (only if not in PATH)
            if 'ninja_path' in self.config and self.config['ninja_path'] != 'ninja':
                ninja_path = self.config['ninja_path'].replace('\\', '/')
                ninja_dir = Path(ninja_path).parent
                ninja_dir_str = str(ninja_dir).replace('\\', '/')
                f.write(f'\n# Ensure CMake can find Ninja\n')
                f.write(f'set(ENV{{PATH}} "$ENV{{PATH}};{ninja_dir_str}")\n')

            # Eigen3 configuration
            if 'eigen3_include' in self.config:
                eigen3_include = self.config['eigen3_include'].replace('\\', '/')
                f.write(f'\n# Eigen3 configuration\n')
                f.write(f'set(EIGEN3_INCLUDE_DIR "{eigen3_include}")\n')
            if 'eigen3_config' in self.config:
                eigen3_config = self.config['eigen3_config'].replace('\\', '/')
                f.write(f'set(EIGEN3_DIR "{eigen3_config}")\n')

            # SIMD library configuration
            if 'simd_include' in self.config:
                simd_include = self.config['simd_include'].replace('\\', '/')
                f.write(f'\n# SIMD library configuration\n')
                f.write(f'set(SIMD_INCLUDE_DIR "{simd_include}")\n')
            if 'simd_config' in self.config:
                simd_config = self.config['simd_config'].replace('\\', '/')
                f.write(f'set(SIMD_DIR "{simd_config}")\n')

        print(f"  {Colors.GREEN}[OK] Generated {self.cmake_cache}{Colors.ENDC}")

    def generate_build_scripts(self):
        """Generate build scripts"""
        if self.system == "Windows":
            compilers = self.config.get('compilers', {})

            if 'msvc' in compilers:
                script_path = Path("build_msvc.bat")
                content = self.generate_windows_msvc_script(compilers['msvc'])
                with open(script_path, 'w') as f:
                    f.write(content)
                print(f"  {Colors.GREEN}[OK] Generated {script_path}{Colors.ENDC}")

            if 'msys2' in compilers:
                script_path = Path("build_msys2.bat")
                content = self.generate_windows_msys2_script(compilers['msys2'])
                with open(script_path, 'w') as f:
                    f.write(content)
                print(f"  {Colors.GREEN}[OK] Generated {script_path}{Colors.ENDC}")
        else:
            # Linux system
            script_path = Path("build.sh")
            content = self.generate_linux_script()
            with open(script_path, 'w') as f:
                f.write(content)
            os.chmod(script_path, 0o755)
            print(f"  {Colors.GREEN}[OK] Generated {script_path}{Colors.ENDC}")

    def generate_windows_msvc_script(self, msvc_info):
        """Generate Windows MSVC build script using config/user_paths.cmake"""
        vcvars_path = msvc_info['vcvars_path']

        return f"""@echo off
echo [INFO] Using MSVC configuration (from config/user_paths.cmake)

REM Set up MSVC environment from config/user_paths.cmake
call "{vcvars_path}" x64
if errorlevel 1 (
    echo [ERROR] Failed to set up MSVC environment
    echo [INFO] Please check vcvars_path in config/user_paths.cmake
    exit /b 1
)

echo [INFO] Building project with Windows MSVC Release preset...

REM Configure and build with CMake preset
cmake --preset windows-msvc-release
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    echo [INFO] Check that all paths in config/user_paths.cmake are correct
        exit /b 1
)

cmake --build build/windows-msvc-release --parallel
if errorlevel 1 (
    echo [ERROR] Build failed
        exit /b 1
)

echo [OK] MSVC build completed successfully!
echo [INFO] Test executables are located in: build/windows-msvc-release/tests/unit_tests/
"""

    def generate_windows_msys2_script(self, msys2_info):
        """Generate Windows MSYS2 build script using config/user_paths.cmake"""
        msys2_path = msys2_info['msys2_path']
        environment = msys2_info['environment']

        # Add Ninja path to environment if available
        ninja_path_line = ""
        if 'ninja_path' in self.config:
            ninja_dir = Path(self.config['ninja_path']).parent
            ninja_path_line = f"set PATH={ninja_dir};%PATH%\n"

        return f"""@echo off
echo [INFO] Using MSYS2 GCC configuration (from config/user_paths.cmake)

REM Set up MSYS2 environment from config/user_paths.cmake
set PATH={msys2_path}\\{environment}\\bin;%PATH%
set MSYSTEM={environment}
{ninja_path_line}echo [INFO] Using MSYS2: {msys2_path}
echo [INFO] Using environment: {environment}

echo [INFO] Building project with Windows MSYS2 Release preset...

REM Configure and build with CMake preset
cmake --preset windows-msys2-release
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    echo [INFO] Check that all paths in config/user_paths.cmake are correct
        exit /b 1
)

cmake --build build/windows-msys2-release --parallel
if errorlevel 1 (
    echo [ERROR] Build failed
        exit /b 1
)

echo [OK] MSYS2 build completed successfully!
echo [INFO] Test executables are located in: build/windows-msys2-release/tests/unit_tests/
"""

    def generate_linux_script(self):
        """Generate Linux build script"""
        return f"""#!/bin/bash
echo [INFO] Using GCC - Simple version

echo [INFO] Building project...
mkdir -p build

# Configure with CMake preset
cmake --preset linux-gcc-release
if [ $? -ne 0 ]; then
    echo [ERROR] CMake configuration failed
    exit 1
fi

# Build the project
cmake --build build/linux-gcc-release --parallel
if [ $? -ne 0 ]; then
    echo [ERROR] Build failed
    exit 1
fi

echo [OK] Linux build completed successfully!
echo [INFO] Test executables are located in: build/linux-gcc-release/tests/unit_tests/
"""

def main():
    print(f"{Colors.GREEN}Starting simple configuration wizard...{Colors.ENDC}")
    configurator = SmartConfigurator()
    success = configurator.run()

    if success:
        print(f"\n{Colors.GREEN}[OK] Configuration completed successfully!{Colors.ENDC}")
        print(f"{Colors.CYAN}Next steps:{Colors.ENDC}")

        # Show appropriate next steps based on system
        if configurator.system == "Windows":
            print(f"  - Run build_msvc.bat (Windows MSVC)")
            print(f"  - Run build_msys2.bat (Windows MSYS2)")
        elif configurator.system == "Linux":
            print(f"  - Run ./build.sh (Linux GCC)")
            print(f"  - Or: chmod +x build.sh && ./build.sh")
        else:
            print(f"  - Run appropriate build script for your system")
    else:
        print(f"{Colors.FAIL}[ERROR] Configuration failed!{Colors.ENDC}")

    return success

if __name__ == "__main__":
    import platform
    success = main()
    sys.exit(0 if success else 1)