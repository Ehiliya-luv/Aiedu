#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合测试脚本：验证reward_new实现可以与main.py集成运行
"""
import sys
import os

# 添加项目路径以支持从test文件夹运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有模块导入"""
    print("=" * 80)
    print("测试1：模块导入")
    print("=" * 80)
    
    try:
        from utils.reward import compute_basic_reward, compute_advanced_reward
        from utils.reward_new import TrainableRewardWeights
        
        print("✓ Reward相关模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_computation():
    """测试reward计算"""
    print("\n" + "=" * 80)
    print("测试2：Reward计算")
    print("=" * 80)
    
    from utils.reward import compute_basic_reward, compute_advanced_reward
    
    test_cases = [
        {
            "name": "完全相同文本",
            "original": "Patient takes 10mg aspirin daily.",
            "revised": "Patient takes 10mg aspirin daily.",
            "expected_range": (0.95, 1.0)
        },
        {
            "name": "小幅修改",
            "original": "Patient takes 10mg aspirin daily.",
            "revised": "Patient takes 10mg aspirin twice daily.",
            "expected_range": (0.5, 1.0)
        },
        {
            "name": "剂量改变",
            "original": "Patient takes 10mg aspirin daily.",
            "revised": "Patient takes 20mg aspirin daily.",
            "expected_range": (0.4, 1.0)
        },
        {
            "name": "医学文本 - 症状",
            "original": "Patient with fever and headache",
            "revised": "Patient with fever, headache and nausea",
            "expected_range": (0.5, 1.0)
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {test_case['name']}")
        print(f"  原文: {test_case['original']}")
        print(f"  修改: {test_case['revised']}")
        
        try:
            basic = compute_basic_reward(test_case['original'], test_case['revised'])
            advanced = compute_advanced_reward(test_case['original'], test_case['revised'])
            
            print(f"  Basic Reward: {basic:.4f}")
            print(f"  Advanced Reward: {advanced:.4f}")
            
            # 检查范围
            min_val, max_val = test_case['expected_range']
            if not (min_val <= basic <= 1.0):
                print(f"  ⚠ Basic Reward超出范围: {min_val} <= {basic:.4f} <= 1.0")
                all_passed = False
            if not (min_val <= advanced <= 1.0):
                print(f"  ⚠ Advanced Reward超出范围: {min_val} <= {advanced:.4f} <= 1.0")
                all_passed = False
            
            print("  ✓ 通过")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_trainable_weights():
    """测试可训练权重模块"""
    print("\n" + "=" * 80)
    print("测试3：可训练权重模块")
    print("=" * 80)
    
    try:
        import torch
        from utils.reward_new import TrainableRewardWeights
        
        # 创建权重模块
        weights = TrainableRewardWeights(initial_e=0.5, initial_t=0.5)
        print("✓ TrainableRewardWeights创建成功")
        
        # 测试forward pass
        r_e = torch.tensor([0.8, 0.6, 0.9])
        r_t = torch.tensor([0.7, 0.8, 0.85])
        
        reward = weights(r_e, r_t)
        print(f"✓ Forward pass成功，输出: {reward}")
        
        # 检查权重
        w_e, w_t = weights.get_weights()
        print(f"✓ 当前权重 - lambda_e: {w_e:.4f}, lambda_t: {w_t:.4f}")
        
        assert abs((w_e + w_t) - 1.0) < 1e-6, "权重应该和为1"
        print("✓ 权重归一化检查通过")
        
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpo_integration():
    """测试与GRPO的集成"""
    print("\n" + "=" * 80)
    print("测试4：Reward函数签名检查")
    print("=" * 80)
    
    try:
        from utils.reward import compute_basic_reward, compute_advanced_reward
        
        # 检查reward函数是否有正确的签名
        import inspect
        
        sig_basic = inspect.signature(compute_basic_reward)
        sig_advanced = inspect.signature(compute_advanced_reward)
        
        print(f"✓ compute_basic_reward 签名: {sig_basic}")
        print(f"✓ compute_advanced_reward 签名: {sig_advanced}")
        
        # 检查关键参数
        basic_params = set(sig_basic.parameters.keys())
        advanced_params = set(sig_advanced.parameters.keys())
        
        required_params = {'original', 'revised'}
        
        if required_params.issubset(basic_params):
            print("✓ compute_basic_reward 有所需参数")
        else:
            print(f"❌ compute_basic_reward 缺少参数: {required_params - basic_params}")
            return False
        
        if required_params.issubset(advanced_params):
            print("✓ compute_advanced_reward 有所需参数")
        else:
            print(f"❌ compute_advanced_reward 缺少参数: {required_params - advanced_params}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  Reward实现综合测试".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    results = {
        "导入测试": test_imports(),
        "Reward计算": test_reward_computation(),
        "可训练权重": test_trainable_weights(),
        "函数签名检查": test_grpo_integration(),
    }
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！可以从main.py开始运行整个流程。")
        print("\n运行建议的命令：")
        print("  # SFT + RL 模式（使用新的advanced reward）")
        print("  python main.py --mode sft+rl --reward-type advanced")
        print("\n  # 仅RL模式")
        print("  python main.py --mode rl --reward-type advanced")
        print("\n  # 使用基础reward")
        print("  python main.py --mode rl --reward-type basic")
    else:
        print("❌ 某些测试失败，请检查上面的错误信息。")
        sys.exit(1)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
