# Debug script to show XML joint loading vs actual usage
# IMPORTANT: Import isaacgym modules BEFORE torch to avoid import errors
try:
    # Import isaacgym first
    import isaacgym
    from isaacgym import gymapi
    print("IsaacGym imported successfully")
except ImportError as e:
    print(f"Warning: Could not import IsaacGym: {e}")
    print("Continuing without IsaacGym...")

# Now import torch and other modules
import torch
import sys
import os
from pathlib import Path

# Add the project root to Python path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from amp_rsl_rl.utils.motion_lib_taihu import MotionLibTaihu
    MOTION_LIB_AVAILABLE = True
except ImportError as e:
    print(f"Could not import MotionLibTaihu: {e}")
    MOTION_LIB_AVAILABLE = False

def debug_joint_dof_mapping(motion_file, mjcf_file):
    """
    Debug function to show what joints/DOFs are loaded from XML vs actually used
    """
    if not MOTION_LIB_AVAILABLE:
        print("MotionLibTaihu not available. Cannot proceed with debugging.")
        return None
        
def debug_xml_structure(mjcf_file):
    """
    Parse XML directly to show joint structure without requiring MotionLib
    """
    print("="*80)
    print("XML STRUCTURE DEBUG (Direct XML Parsing)")
    print("="*80)
    
    try:
        import xml.etree.ElementTree as ET
        
        if not os.path.exists(mjcf_file):
            print(f"Error: MJCF file not found: {mjcf_file}")
            return
            
        tree = ET.parse(mjcf_file)
        root = tree.getroot()
        
        print(f"MJCF file: {mjcf_file}")
        print(f"Root element: {root.tag}")
        print()
        
        # Find all joints
        joints = root.findall(".//joint")
        print(f"1. JOINTS FROM XML ({len(joints)}):")
        print("-" * 40)
        for i, joint in enumerate(joints):
            name = joint.get('name', f'unnamed_{i}')
            joint_type = joint.get('type', 'hinge')
            axis = joint.get('axis', '0 0 1')
            range_attr = joint.get('range', 'unlimited')
            print(f"  [{i:2d}] {name:20s} | type: {joint_type:8s} | axis: {axis:8s} | range: {range_attr}")
        
        print()
        
        # Find all bodies
        bodies = root.findall(".//body")
        print(f"2. BODIES FROM XML ({len(bodies)}):")
        print("-" * 40)
        for i, body in enumerate(bodies):
            name = body.get('name', f'unnamed_body_{i}')
            pos = body.get('pos', '0 0 0')
            print(f"  [{i:2d}] {name:25s} | pos: {pos}")
        
        print()
        
        # Find actuators
        actuators = root.findall(".//actuator")
        if actuators:
            print(f"3. ACTUATORS FROM XML ({len(actuators)}):")
            print("-" * 40)
            for i, actuator in enumerate(actuators):
                name = actuator.get('name', f'unnamed_actuator_{i}')
                joint_ref = actuator.get('joint', 'none')
                print(f"  [{i:2d}] {name:20s} | controls joint: {joint_ref}")
        else:
            motors = root.findall(".//motor")
            if motors:
                print(f"3. MOTORS FROM XML ({len(motors)}):")
                print("-" * 40)
                for i, motor in enumerate(motors):
                    name = motor.get('name', f'unnamed_motor_{i}')
                    joint_ref = motor.get('joint', 'none')
                    print(f"  [{i:2d}] {name:20s} | controls joint: {joint_ref}")
        
        print()
        
        # Check for specific patterns
        print("4. DOF ANALYSIS:")
        print("-" * 40)
        print(f"Total joints found: {len(joints)}")
        
        # Count moveable joints (exclude fixed joints)
        moveable_joints = [j for j in joints if j.get('type', 'hinge') != 'fixed']
        print(f"Moveable joints: {len(moveable_joints)}")
        
        # Expected DOF based on filename
        if "12dof" in mjcf_file.lower():
            expected_dof = 12
            print(f"Expected DOF (from filename): {expected_dof}")
            if len(moveable_joints) != expected_dof:
                print(f"  ⚠️  MISMATCH: Expected {expected_dof}, found {len(moveable_joints)} moveable joints")
            else:
                print(f"  ✅ DOF count matches filename!")
        
        return {
            'joints': [(j.get('name'), j.get('type', 'hinge')) for j in joints],
            'bodies': [b.get('name') for b in bodies],
            'moveable_joints': len(moveable_joints),
            'total_joints': len(joints)
        }
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

def debug_motion_lib_mapping(motion_file, mjcf_file):
    """
    Debug function to show what joints/DOFs are loaded from XML vs actually used
    """
    print("="*80)
    print("JOINT/DOF LOADING DEBUG")
    print("="*80)
    
    # Initialize MotionLib
    device = torch.device("cpu")
    motion_lib = MotionLibTaihu(
        motion_file=motion_file,
        device=device,
        mjcf_file=mjcf_file,
        extend_hand=False,
        extend_head=False,
        sim_timestep=1.0/60.0,  # default timestep
    )
    
    # Enable debug mode if available
    if hasattr(motion_lib, "set_joint_mapping_mode"):
        motion_lib.set_joint_mapping_mode(use_mapping=True, debug=True)
    
    print(f"Motion file: {motion_file}")
    print(f"MJCF file: {mjcf_file}")
    print()
    
    # 1. Show XML joint structure
    print("1. JOINTS/DOFS FROM XML:")
    print("-" * 40)
    
    # Try to access mesh parser info
    if hasattr(motion_lib, 'mesh_parsers'):
        parser = motion_lib.mesh_parsers
        
        # Joint names from XML
        if hasattr(parser, 'joint_names'):
            xml_joints = parser.joint_names
            print(f"XML Joint Names ({len(xml_joints)}):")
            for i, joint in enumerate(xml_joints):
                print(f"  [{i:2d}] {joint}")
        
        # DOF names from XML  
        if hasattr(parser, 'dof_names'):
            xml_dofs = parser.dof_names
            print(f"\nXML DOF Names ({len(xml_dofs)}):")
            for i, dof in enumerate(xml_dofs):
                print(f"  [{i:2d}] {dof}")
                
        # Body/model names
        if hasattr(parser, 'model_names'):
            model_names = parser.model_names
            print(f"\nXML Model/Body Names ({len(model_names)}):")
            for i, name in enumerate(model_names):
                print(f"  [{i:2d}] {name}")
        elif hasattr(parser, 'body_names'):
            body_names = parser.body_names
            print(f"\nXML Body Names ({len(body_names)}):")
            for i, name in enumerate(body_names):
                print(f"  [{i:2d}] {name}")
    
def debug_motion_lib_mapping(motion_file, mjcf_file):
    """
    Debug MotionLib loading if available
    """
    if not MOTION_LIB_AVAILABLE:
        print("MotionLibTaihu not available. Skipping MotionLib debug.")
        return None
        
    print("="*80)
    print("MOTION LIB DEBUG")
    print("="*80)
    
    # Initialize MotionLib
    device = torch.device("cpu")
    motion_lib = MotionLibTaihu(
        motion_file=motion_file,
        device=device,
        mjcf_file=mjcf_file,
        extend_hand=False,
        extend_head=False,
        sim_timestep=1.0/60.0,  # default timestep
    )
    
    # Enable debug mode if available
    if hasattr(motion_lib, "set_joint_mapping_mode"):
        motion_lib.set_joint_mapping_mode(use_mapping=True, debug=True)
    
    print(f"Motion file: {motion_file}")
    print(f"MJCF file: {mjcf_file}")
    print()
    
    # 1. Show XML joint structure from motion lib perspective
    print("1. MOTION LIB PARSED STRUCTURE:")
    print("-" * 40)
    
    # Try to access mesh parser info
    if hasattr(motion_lib, 'mesh_parsers'):
        parser = motion_lib.mesh_parsers
        
        # Joint names from XML
        if hasattr(parser, 'joint_names'):
            xml_joints = parser.joint_names
            print(f"Parsed Joint Names ({len(xml_joints)}):")
            for i, joint in enumerate(xml_joints):
                print(f"  [{i:2d}] {joint}")
        
        # DOF names from XML  
        if hasattr(parser, 'dof_names'):
            xml_dofs = parser.dof_names
            print(f"\nParsed DOF Names ({len(xml_dofs)}):")
            for i, dof in enumerate(xml_dofs):
                print(f"  [{i:2d}] {dof}")
                
        # Body/model names
        if hasattr(parser, 'model_names'):
            model_names = parser.model_names
            print(f"\nParsed Model/Body Names ({len(model_names)}):")
            for i, name in enumerate(model_names):
                print(f"  [{i:2d}] {name}")
        elif hasattr(parser, 'body_names'):
            body_names = parser.body_names
            print(f"\nParsed Body Names ({len(body_names)}):")
            for i, name in enumerate(body_names):
                print(f"  [{i:2d}] {name}")
    
    print()
    
    # 2. Try to load sample motion data
    print("2. SAMPLE MOTION DATA:")
    print("-" * 40)
    
    try:
        # Create dummy skeleton trees
        node_names = getattr(motion_lib.mesh_parsers, "model_names", None) or getattr(motion_lib.mesh_parsers, "body_names", [])
        
        if not node_names:
            print("No node names found, cannot proceed with motion loading")
            return motion_lib
            
        class DummySkel:
            def __init__(self, names): 
                self.node_names = names
        
        skeleton_trees = [DummySkel(node_names)]
        gender_betas = [torch.zeros(17)]
        limb_weights = [1.0] * len(node_names)
        
        motion_lib.load_motions(
            skeleton_trees=skeleton_trees,
            gender_betas=gender_betas,
            limb_weights=limb_weights,
            random_sample=False,
            start_idx=0,
            max_len=1,  # Just load 1 frame for debugging
        )
        
        # Get a sample motion state
        mids = motion_lib.sample_motions(1)
        t0 = motion_lib.sample_time(mids, truncate_time=0.0)
        motion_state = motion_lib.get_motion_state(mids, t0)
        
        print("Motion State Keys:", list(motion_state.keys()))
        
        for key, value in motion_state.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                
                # Show DOF dimensions specifically
                if key == "dof_pos":
                    print(f"    -> DOF positions: {value.shape[-1]} DOFs")
                    if value.shape[-1] <= 20:  # Only print if reasonable number
                        print(f"    -> Values: {value.flatten().tolist()}")
                elif key == "dof_vel":
                    print(f"    -> DOF velocities: {value.shape[-1]} DOFs")
            else:
                print(f"  {key}: {type(value)}")
        
    except Exception as e:
        print(f"Error loading motions: {e}")
        import traceback
        traceback.print_exc()
    
    return motion_lib


# Example usage and main function
def main():
    print("JOINT/DOF DEBUG TOOL")
    print("=" * 80)
    
    # Default paths from your error - adjust these to your actual files
    mjcf_file = "/home/dy/dy/code/unitree_ti/assert/ti5/ti5_12dof.xml"
    motion_file = None  # Will ask user to provide
    
    if len(sys.argv) > 1:
        mjcf_file = sys.argv[1]
    if len(sys.argv) > 2:
        motion_file = sys.argv[2]
    
    print(f"Using MJCF file: {mjcf_file}")
    
    # First, try direct XML parsing (always works)
    xml_info = debug_xml_structure(mjcf_file)
    
    # Then try MotionLib if motion file is provided
    if motion_file and os.path.exists(motion_file):
        print(f"Using motion file: {motion_file}")
        motion_lib = debug_motion_lib_mapping(motion_file, mjcf_file)
    else:
        print("\nTo also debug MotionLib loading:")
        print("python debug_joint.py <mjcf_file> <motion_file>")
        print("Example:")
        print(f"python debug_joint.py {mjcf_file} /path/to/your/motion_data.npy")
    
    print("=" * 80)
if __name__ == "__main__":
    main()