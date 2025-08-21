import os
import bpy
import subprocess

def decimate_mesh(input_path, output_path, decimate_ratio=0.1):
    """
    使用Blender的Python API对STL文件进行减面处理
    :param input_path: 输入的STL文件路径
    :param output_path: 输出的STL文件路径
    :param decimate_ratio: 减面比例（0到1之间，越小减面越多）
    """
    # 清空Blender场景
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 导入STL文件
    bpy.ops.import_mesh.stl(filepath=input_path)

    # 获取导入的物体
    obj = bpy.context.selected_objects[0]

    # 添加减面修饰符
    bpy.ops.object.modifier_add(type='DECIMATE')
    decimate_modifier = obj.modifiers["Decimate"]
    decimate_modifier.ratio = decimate_ratio

    # 应用减面修饰符
    bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)

    # 导出处理后的STL文件
    bpy.ops.export_mesh.stl(filepath=output_path)

def process_stl_files(folder_path, decimate_ratio=0.1):
    """
    处理文件夹中的所有STL文件
    :param folder_path: 包含STL文件的文件夹路径
    :param decimate_ratio: 减面比例
    """
    # 获取文件夹中所有STL文件
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]
    total_files = len(stl_files)

    for i, stl_file in enumerate(stl_files, start=1):
        input_path = os.path.join(folder_path, stl_file)
        output_path = input_path  # 覆盖原文件

        # 调用Blender进行减面处理
        decimate_mesh(input_path, output_path, decimate_ratio)

        # 显示进度
        print(f"Processed {i}/{total_files} files: {stl_file}")

if __name__ == "__main__":
    # 设置文件夹路径和减面比例
    folder_path = "E:\\T170-V2.1-A0-URDF-A\\meshes2\\"  # 替换为你的STL文件夹路径
    decimate_ratio = 0.1  # 减面比例，可以根据需要调整

    # 启动Blender并运行脚本
    blender_path = "C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender-launcher.exe"  # 替换为你的Blender可执行文件路径
    script_path = os.path.abspath(__file__)  # 当前脚本路径

    # 使用Blender的命令行模式运行脚本
    subprocess.run([blender_path, "--background", "--python", script_path, "--", folder_path, str(decimate_ratio)])