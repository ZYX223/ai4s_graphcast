// 最近偶数舍入（RNE）教学示例，适合直接分享给同事。
// 编译：g++ -std=c++11 -O2 -Wall -Wextra -Wpedantic rounding_demo.cpp -o rounding_demo
// 运行：./rounding_demo
//
// 本例在 CPU 上解释位运算；核心表达式与实验中的 HIP GPU kernel 相同。
// 最后的 A×I 是 CPU 数值模拟，不调用 GPU，也不模拟 XF32 的累加指令。

#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static_assert(sizeof(float) == sizeof(std::uint32_t), "Requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24 &&
              std::numeric_limits<float>::max_exponent == 128,
              "Requires IEEE 754 binary32");

// 按位复制，不是数值转换。例如 1.0f 的编码是 0x3f800000，不能用 (uint32_t)x。
// memcpy 是标准 C++ 的安全写法；GPU 版本对应 __float_as_uint(x)。
std::uint32_t float_to_bits(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

// GPU 版本对应 __uint_as_float(bits)，得到的存储类型仍然是 FP32。
float bits_to_float(std::uint32_t bits) {
    float x;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

// 对照组：直接清掉小数低 13 位。有限数的绝对值只会减小或保持不变。
float truncate_to_10_bits(float x) {
    std::uint32_t bits = float_to_bits(x);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits &= 0xffffe000u;
    }
    return bits_to_float(bits);
}

// 目标：显式小数从 23 位减至 10 位，采用最近偶数舍入。
// 数据仍放在 float 中，符号位、指数位的格式和位宽不变。
// 注意：舍入进位可能改变指数；很大的有限数也可能舍入为无穷大。
float round_to_10_bits_rne(float x) {
    std::uint32_t bits = float_to_bits(x);

    // 指数全为 1 表示 NaN 或 ±Inf。这些编码直接保留，包括 NaN payload。
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        // 需要清掉 23-10=13 位，因此一步的编码间隔是 2^13=8192。
        // 中点是 4096。先加 4095，再根据保留部分最低位补 0 或 1：
        //   低13位 < 4096：不会进位；> 4096：一定进位。
        //   低13位 = 4096：保留位为0时不进位，为1时进位，保证结果为偶数。
        // 这里的“偶数”是保留编码的最低位为0，不要求原值是整数。
        const std::uint32_t retained_lsb = (bits >> 13) & 1u;
        bits = (bits + 0x00000fffu + retained_lsb) & 0xffffe000u;
    }
    return bits_to_float(bits);
}

// 输出：符号1位 | 指数8位 | 保留小数10位 | 将清掉的小数13位。
std::string show_bits(float x) {
    const std::string s = std::bitset<32>(float_to_bits(x)).to_string();
    return s.substr(0, 1) + " | " + s.substr(1, 8) + " | " +
           s.substr(9, 10) + " | " + s.substr(19, 13);
}

void require_same_bits(float actual, float expected, const std::string& label) {
    if (float_to_bits(actual) != float_to_bits(expected)) {
        throw std::runtime_error("结果不符合预期：" + label);
    }
}

// 逐步展开一个普通有限值，方便对照核心公式。
void explain(float x, const char* label) {
    const std::uint32_t bits = float_to_bits(x);
    const std::uint32_t low = bits & 0x1fffu;
    const std::uint32_t lsb = (bits >> 13) & 1u;
    const std::uint32_t bias = 0xfffu + lsb;
    std::cout << "\n" << label << "：x = " << x << '\n'
              << "  输入位       " << show_bits(x) << '\n'
              << "  低13位 = " << low << "，保留部分最低位 = " << lsb
              << "，加数 = " << bias << '\n'
              << "  低13位 + 加数 = " << low + bias
              << "，是否达到8192并进位：" << (low + bias >= 8192 ? "是" : "否") << '\n'
              << "  舍入后       " << show_bits(round_to_10_bits_rne(x)) << '\n'
              << "  直接截断 = " << truncate_to_10_bits(x)
              << "，最近偶数舍入 = " << round_to_10_bits_rne(x) << '\n';
}

struct Example {
    const char* name;
    float input;
    float expected_trunc;
    float expected_rne;
};

void examples() {
    // [1,2) 区间，保留10位小数后的相邻值间隔为 2^-10=0.0009765625。
    // 以下十进制常量都是精确的二进制分数，FP32 能精确表示这些输入。
    const Example cases[] = {
        {"near_lower",    1.000244140625f,  1.0f,          1.0f},
        {"near_upper",    1.000732421875f,  1.0f,          1.0009765625f},
        {"tie_even",      1.00048828125f,   1.0f,          1.0f},
        {"tie_odd",       1.00146484375f,   1.0009765625f, 1.001953125f},
        {"negative",     -1.000732421875f, -1.0f,         -1.0009765625f},
        {"neg_tie_even", -1.00048828125f,  -1.0f,         -1.0f},
        {"neg_tie_odd",  -1.00146484375f,  -1.0009765625f,-1.001953125f},
        {"carry_exponent",1.999755859375f,  1.9990234375f, 2.0f},
        {"already_exact", 1.5f,             1.5f,          1.5f},
    };
    std::cout << "示例对照：original=原值，truncate=向零截断，RNE=最近偶数舍入\n"
              << std::left << std::setw(18) << "case"
              << std::setw(21) << "original" << std::setw(21) << "truncate"
              << "RNE\n";
    for (const auto& item : cases) {
        const float trunc = truncate_to_10_bits(item.input);
        const float rne = round_to_10_bits_rne(item.input);
        require_same_bits(trunc, item.expected_trunc, item.name);
        require_same_bits(rne, item.expected_rne, item.name);
        // 已经舍入过的普通值，再截断或再次舍入不会改变其编码。
        require_same_bits(truncate_to_10_bits(rne), rne, "RNE后再截断");
        require_same_bits(round_to_10_bits_rne(rne), rne, "RNE重复执行");
        std::cout << std::setw(18) << item.name << std::setw(21) << item.input
                  << std::setw(21) << trunc << rne << '\n';
    }
}

void special_values() {
    // 比较编码而不是 float ==：NaN 不等于自身，+0/-0 数值相等但编码不同。
    const std::uint32_t unchanged[] = {
        0x00000000u, 0x80000000u,  // +0, -0
        0x7f800000u, 0xff800000u,  // +Inf, -Inf
        0x7fc01234u, 0xffc05678u,  // 带不同 payload 的正负 quiet NaN
    };
    for (std::uint32_t bits : unchanged) {
        const float x = bits_to_float(bits);
        require_same_bits(round_to_10_bits_rne(x), x, "特殊编码保留");
        require_same_bits(truncate_to_10_bits(x), x, "特殊编码截断对照");
    }
    require_same_bits(round_to_10_bits_rne(bits_to_float(0x7f7fffffu)),
                      bits_to_float(0x7f800000u), "最大有限正数舍入为Inf");
    require_same_bits(round_to_10_bits_rne(bits_to_float(0xff7fffffu)),
                      bits_to_float(0xff800000u), "最大有限负数舍入为-Inf");
    require_same_bits(round_to_10_bits_rne(bits_to_float(0x00000001u)),
                      bits_to_float(0x00000000u), "最小正次正规数舍入为+0");
    require_same_bits(round_to_10_bits_rne(bits_to_float(0x80000001u)),
                      bits_to_float(0x80000000u), "最小负次正规数舍入为-0");
    std::cout << "\n特殊值检查通过：±0、±Inf、NaN payload、溢出和极小值。\n";
}

// 2×2 CPU 矩阵乘：只演示输入预舍入，不声称复现实际GPU的累加顺序。
void matmul2(const float a[4], const float b[4], float c[4]) {
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            c[row * 2 + col] = 0.0f;
            for (int k = 0; k < 2; ++k) {
                c[row * 2 + col] += a[row * 2 + k] * b[k * 2 + col];
            }
        }
    }
}

void identity_demo() {
    const float a[4] = {1.000732421875f, -1.000732421875f,
                        1.00048828125f, 1.00146484375f};
    const float identity[4] = {1, 0, 0, 1};
    float at[4], ar[4], it[4], ir[4], full[4], trunc[4], rne[4];
    for (int i = 0; i < 4; ++i) {
        at[i] = truncate_to_10_bits(a[i]);
        ar[i] = round_to_10_bits_rne(a[i]);
        it[i] = truncate_to_10_bits(identity[i]);
        ir[i] = round_to_10_bits_rne(identity[i]);
    }
    matmul2(a, identity, full);
    matmul2(at, it, trunc);
    matmul2(ar, ir, rne);
    double trunc_mae = 0.0, rne_mae = 0.0;
    for (int i = 0; i < 4; ++i) {
        require_same_bits(full[i], a[i], "FP32 A乘单位阵");
        require_same_bits(trunc[i], at[i], "截断输入后A乘单位阵");
        require_same_bits(rne[i], ar[i], "RNE输入后A乘单位阵");
        trunc_mae += std::abs(double(trunc[i]) - double(a[i])) / 4;
        rne_mae += std::abs(double(rne[i]) - double(a[i])) / 4;
    }
    std::cout << "\nA × I：CPU输入精度模拟，I中的0和1能精确表示。\n";
    const float* matrices[] = {full, trunc, rne};
    const char* labels[] = {"FP32结果（等于A）", "先截断输入", "先最近偶数舍入输入"};
    for (int m = 0; m < 3; ++m) {
        std::cout << labels[m] << ":\n";
        for (int row = 0; row < 2; ++row) {
            std::cout << "  [ " << std::setw(20) << matrices[m][row * 2]
                      << ", " << std::setw(20) << matrices[m][row * 2 + 1] << " ]\n";
        }
    }
    std::cout << "相对原始A的MAE：截断=" << trunc_mae << "，RNE=" << rne_mae
              << "，本例降低=" << (1.0 - rne_mae / trunc_mae) * 100 << "%\n";
}

int main() {
    try {
        std::cout << std::setprecision(15);
        examples();
        std::cout << "\n位布局：符号1位 | 指数8位 | 保留小数10位 | 清除小数13位\n";
        explain(1.000732421875f, "更接近上邻值");
        explain(1.00048828125f, "恰好中点，保留部分为偶数");
        explain(1.00146484375f, "恰好中点，保留部分为奇数");
        explain(-1.000732421875f, "负数同样取最近值");
        special_values();
        identity_demo();
        std::cout << "\n全部示例及检查通过。GPU实现可将两个memcpy助手换成对应位转换内建函数。\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
