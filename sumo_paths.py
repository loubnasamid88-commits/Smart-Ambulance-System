from pathlib import Path
import os
import shutil
import sys


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SUMO_CONFIG = BASE_DIR / "marrakech_sumo" / "marrakech_sumo" / "marrakech.sumocfg"


def default_sumo_home():
    if "SUMO_HOME" in os.environ:
        return Path(os.environ["SUMO_HOME"])

    common_paths = [
        Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
        Path(r"C:\Program Files\Eclipse\Sumo"),
    ]
    for path in common_paths:
        if path.exists():
            return path

    home = Path.home()
    search_roots = [
        home / "Downloads",
        home / "Documents",
        home,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.glob("sumo*/sumo-*"):
            if (candidate / "bin" / tool_name("sumo-gui")).exists():
                return candidate
        for candidate in root.glob("**/sumo-gui.exe"):
            if candidate.parent.name.lower() == "bin":
                return candidate.parent.parent

    return None


def tool_name(name):
    return f"{name}.exe" if os.name == "nt" else name


def find_sumo_tool(name):
    env_name = name.upper().replace("-", "_")
    if env_name in os.environ:
        return Path(os.environ[env_name])

    home = default_sumo_home()
    if home:
        candidate = home / "bin" / tool_name(name)
        if candidate.exists():
            return candidate

    found = shutil.which(tool_name(name)) or shutil.which(name)
    if found:
        return Path(found)

    if home:
        return home / "bin" / tool_name(name)
    return Path(tool_name(name))


def sumo_tools_path():
    home = default_sumo_home()
    if home:
        return home / "tools"
    random_trips = find_sumo_tool("randomTrips.py")
    if random_trips.exists():
        return random_trips.parent
    return None


def ensure_traci_on_path():
    tools = sumo_tools_path()
    if tools and str(tools) not in sys.path:
        sys.path.append(str(tools))


def sumo_binary(gui=False):
    override = os.environ.get("SUMO_BINARY")
    if override:
        return Path(override)
    return find_sumo_tool("sumo-gui" if gui else "sumo")


def sumo_config():
    return Path(os.environ.get("SUMO_CONFIG", DEFAULT_SUMO_CONFIG))
