// 多样本加载测试：不运行训练，仅加载数据集前几个样本，打印 ModelInput 各张量维度。
// 用法: ./ppml_test_dataloader [root] [max_samples]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "ppml/DataLoader.h"
#include "ppml/Model.h"

using namespace ppml;

namespace fs = std::filesystem;

static std::string read_a3m_query_sequence(const std::string& a3m_path) {
    std::ifstream f(a3m_path);
    if (!f) throw std::runtime_error("read_a3m_query_sequence: cannot open " + a3m_path);
    std::string line;
    bool seen_header = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') { seen_header = true; continue; }
        if (seen_header) {
            std::string seq;
            for (char c : line) if (c != '\n' && c != '\r') seq.push_back(c);
            return seq;
        }
    }
    throw std::runtime_error("read_a3m_query_sequence: no sequence found in " + a3m_path);
}

// 与 train.cpp 的 discover_training_set 等价：配对 <U>_alignment.a3m 与 *_<U>_mapping.csv 目录
struct Sample { std::string a3m; std::string csv_dir; std::string uniprot; };
static std::vector<Sample> discover_training_set(const std::string& root) {
    std::vector<Sample> out;
    if (!fs::exists(root)) return out;
    std::map<std::string, std::string> a3m_of;
    for (auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        auto fn = p.filename().string();
        // <U>_alignment.a3m
        const std::string suf = "_alignment.a3m";
        if (fn.size() > suf.size() && fn.compare(fn.size() - suf.size(), suf.size(), suf) == 0) {
            std::string u = fn.substr(0, fn.size() - suf.size());
            a3m_of[u] = p.string();
        }
    }
    for (auto& kv : a3m_of) {
        const std::string& u = kv.first;
        const std::string& a3m = kv.second;
        // 找对应 csv: 目录 <U>_mapping_results 或 文件 *_<U>_mapping.csv
        std::string csv_dir;
        std::string d1 = (fs::path(root) / (u + "_mapping_results")).string();
        if (fs::is_directory(d1)) csv_dir = d1;
        if (csv_dir.empty()) {
            for (auto& e : fs::directory_iterator(root)) {
                if (!e.is_regular_file()) continue;
                auto fn2 = e.path().filename().string();
                const std::string suf2 = "_" + u + "_mapping.csv";
                if (fn2.size() > suf2.size() &&
                    fn2.compare(fn2.size() - suf2.size(), suf2.size(), suf2) == 0) {
                    csv_dir = e.path().parent_path().string(); // 单文件也传目录（loader 会合并同目录所有 *_mapping.csv）
                    break;
                }
            }
        }
        out.push_back({a3m, csv_dir, u});
    }
    return out;
}

template <typename T>
static void dump_shape(const char* name, const T& t) {
    int nd = t.shape().ndim();
    std::cout << "  " << name << ": ndim=" << nd << " dims=[";
    for (int i = 0; i < nd; ++i) {
        if (i) std::cout << ",";
        std::cout << t.shape().dims[i];
    }
    std::cout << "] numel=" << t.numel() << std::endl;
}

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : "data/training_batch_data";
    int max_samples = argc > 2 ? std::atoi(argv[2]) : 3;
    int msa_depth = 128;
    if (const char* pmd = std::getenv("PPML_MSA_DEPTH")) msa_depth = std::atoi(pmd);

    std::cout << "==== 多样本加载测试 (不训练) ====" << std::endl;
    auto samples = discover_training_set(root);
    std::cout << "[dataset] 发现 " << samples.size() << " 个蛋白样本 @ " << root << std::endl;
    if (samples.empty()) return 1;

    PPMLDataLoader loader("", "", msa_depth, 4, 2048);
    int ok = 0, fail = 0;
    for (size_t i = 0; i < samples.size() && (int)i < max_samples; ++i) {
        const auto& s = samples[i];
        std::cout << "\n--- Sample " << (i + 1) << "/" << std::min<size_t>(max_samples, samples.size())
                  << ": " << s.uniprot << " ---" << std::endl;
        std::cout << "  a3m: " << s.a3m << std::endl;
        std::cout << "  csv_dir: " << (s.csv_dir.empty() ? "(none)" : s.csv_dir) << std::endl;
        std::string seq;
        try { seq = read_a3m_query_sequence(s.a3m); }
        catch (const std::exception& e) {
            std::cerr << "  [FAIL] read a3m: " << e.what() << std::endl;
            ++fail; continue;
        }
        std::cout << "  query L=" << seq.length() << std::endl;
        if (seq.length() > 2048) {
            std::cout << "  [SKIP] L>2048 超出 loader 上限" << std::endl;
            ++fail; continue;
        }
        try {
            ModelInput input = loader.load_from_files(s.a3m, seq, s.csv_dir, "", "");
            dump_shape("msa_latent", input.msa_latent);
            dump_shape("msa_full", input.msa_full);
            dump_shape("seq_tokens", input.seq_tokens);
            dump_shape("t1d", input.t1d);
            dump_shape("t2d", input.t2d);
            dump_shape("coords", input.coords);
            dump_shape("true_coords", input.true_coords);
            dump_shape("tor_feat", input.tor_feat);
            ++ok;
            std::cout << "  [OK] 加载成功" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  [FAIL] load: " << e.what() << std::endl;
            ++fail;
        }
    }
    std::cout << "\n==== 结果: ok=" << ok << " fail=" << fail << " ====" << std::endl;
    return fail ? 2 : 0;
}
