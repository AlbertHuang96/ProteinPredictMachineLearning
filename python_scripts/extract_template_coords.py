"""
extract_template_coords.py — 从 RCSB 下载 mmCIF 文件并提取骨干原子坐标

用法：
    python extract_template_coords.py template_domain_names.json
    
    python extract_template_coords.py template_domain_names.json --output_dir ./coords --chain A
    
    python extract_template_coords.py template_domain_names.json --all-chains

JSON 文件格式 (template_domain_names.json):
    {"A": ["8f2h_A", "6xre_M", ...], "B": ["3ts8_B", "4mzr_B", ...]}
    或简单列表: ["1A8O_A", "6VXX_B", ...]
"""

import os
import json
import argparse
import requests
import numpy as np
import Bio.PDB


def load_template_ids(json_path, chain_filter=None, all_chains=False):
    """
    从 JSON 文件读取模板 ID 列表。
    
    支持两种 JSON 格式：
    1. 按链分组的字典: {"A": ["8f2h_A", ...], "B": ["3ts8_B", ...]}
    2. 简单列表:        ["1A8O_A", "6VXX_B", ...]
    
    :param json_path:    JSON 文件路径
    :param chain_filter: 指定链 ID (如 "A")，仅读取该链的 ID；None 且 all_chains=False 时读取所有链
    :param all_chains:   是否读取所有链（与 chain_filter 互斥）
    :return:             (template_ids: list, chain_map: dict 或 None)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # 简单列表格式，无链分组
        return data, None

    if isinstance(data, dict):
        if all_chains:
            # 合并所有链的 ID
            all_ids = []
            for chain_ids in data.values():
                all_ids.extend(chain_ids)
            return all_ids, data

        if chain_filter is not None:
            if chain_filter not in data:
                raise KeyError(
                    f"Chain '{chain_filter}' not found in JSON. "
                    f"Available chains: {list(data.keys())}"
                )
            return data[chain_filter], data

        # 默认：返回所有链（向后兼容）
        all_ids = []
        for chain_ids in data.values():
            all_ids.extend(chain_ids)
        return all_ids, data

    raise TypeError(f"Unexpected JSON type: {type(data)}. Expected list or dict.")


def extract_template_coords(template_ids, output_dir="template_coords"):
    """
    下载 mmCIF 文件并提取 N, CA, C, O 3D 坐标矩阵，
    供模型训练管线使用。
    
    :param template_ids: 字符串列表，如 ["1A8O_A", "6VXX_B"]
    :param output_dir:   原始 CIF 文件和 .npy 坐标数组的保存目录
    :return:             提取结果摘要字典 {template_id: {residue_count, coord_matrix_shape, saved_to}}
    """
    os.makedirs(output_dir, exist_ok=True)
    parser = Bio.PDB.MMCIFParser(QUIET=True)

    extracted_data = {}

    for t_id in template_ids:
        parts = t_id.split("_")
        pdb_code = parts[0].lower()
        target_chain = parts[1] if len(parts) > 1 else "A"

        # 1. 从 RCSB 下载原始 mmCIF 坐标文件
        cif_path = os.path.join(output_dir, f"{pdb_code}.cif")
        if not os.path.exists(cif_path):
            cif_url = f"https://files.rcsb.org/download/{pdb_code}.cif"
            res = requests.get(cif_url)
            if res.status_code == 200:
                with open(cif_path, "wb") as f:
                    f.write(res.content)
            else:
                print(f"Failed to fetch {pdb_code}")
                continue

        # 2. 使用 Biopython 解析坐标
        structure = parser.get_structure(pdb_code, cif_path)

        coords_list = []
        res_indices = []

        for model in structure:
            for chain in model:
                if chain.id != target_chain:
                    continue

                for residue in chain:
                    # 过滤标准氨基酸残基
                    if Bio.PDB.is_aa(residue, standard=True):
                        try:
                            # 提取 N, CA, C, O 原子坐标 (每个残基 4 x 3)
                            n_xyz  = residue['N'].get_coord()
                            ca_xyz = residue['CA'].get_coord()
                            c_xyz  = residue['C'].get_coord()
                            o_xyz  = residue['O'].get_coord()

                            # 追加 shape (4, 3) 的数组
                            res_coords = np.stack([n_xyz, ca_xyz, c_xyz, o_xyz], axis=0)
                            coords_list.append(res_coords)
                            res_indices.append(residue.get_id()[1])
                        except KeyError:
                            # 跳过骨干重原子不完整的残基
                            continue

            # 只处理 CIF 中的第一个 model
            break

        if coords_list:
            # Shape: [num_residues, 4, 3] -> (N_res, N_atoms, XYZ)
            coord_tensor = np.stack(coords_list, axis=0)

            # 保存为 NumPy 文件 (.npy)，训练时快速加载
            npy_path = os.path.join(output_dir, f"{t_id}_coords.npy")
            np.save(npy_path, coord_tensor)

            extracted_data[t_id] = {
                "residue_count": len(res_indices),
                "coord_matrix_shape": list(coord_tensor.shape),
                "saved_to": npy_path
            }
            print(f"[{t_id}] Successfully extracted {coord_tensor.shape} backbone coordinates.")

    return extracted_data


def main():
    parser = argparse.ArgumentParser(
        description="从 RCSB 下载 mmCIF 文件并提取骨干原子坐标 (N, CA, C, O)"
    )
    parser.add_argument(
        "json_file",
        help="JSON 文件路径，包含模板 ID 列表 (支持按链分组)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="template_coords",
        help="输出目录 (默认: template_coords)"
    )
    chain_group = parser.add_mutually_exclusive_group()
    chain_group.add_argument(
        "--chain", "-c",
        default=None,
        help="仅处理指定链的 ID (如 -c A)"
    )
    chain_group.add_argument(
        "--all-chains", "-a",
        action="store_true",
        help="处理 JSON 中所有链的 ID"
    )

    args = parser.parse_args()

    # 读取 JSON 并提取 ID 列表
    print(f"Loading template IDs from: {args.json_file}")
    template_ids, chain_map = load_template_ids(
        args.json_file,
        chain_filter=args.chain,
        all_chains=args.all_chains
    )

    if chain_map and args.chain:
        print(f"Processing chain '{args.chain}': {len(template_ids)} templates")
    elif chain_map:
        print(f"Processing all chains ({list(chain_map.keys())}): {len(template_ids)} templates")
    else:
        print(f"Processing {len(template_ids)} templates (flat list)")

    # 提取坐标
    summary = extract_template_coords(template_ids, output_dir=args.output_dir)

    # 输出摘要
    print(f"\nExtraction Summary ({len(summary)}/{len(template_ids)} succeeded):")
    print(json.dumps(summary, indent=2))

    # 保存摘要到 JSON
    summary_path = os.path.join(args.output_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
