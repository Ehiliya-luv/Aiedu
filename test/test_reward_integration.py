#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试reward_new.py中新的advanced reward计算
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reward import compute_basic_reward, compute_advanced_reward

def test_reward():
    """测试reward计算"""
    print("=" * 80)
    print("测试Reward计算")
    print("=" * 80)
    
    # 测试用例1：完全相同
    original = "Patient takes 10mg aspirin daily."
    revised = "Patient takes 10mg aspirin daily."
    
    print(f"\n测试1：完全相同")
    print(f"  原文: {original}")
    print(f"  修改: {revised}")
    basic = compute_basic_reward(original, revised)
    advanced = compute_advanced_reward(original, revised)
    print(f"  Basic Reward: {basic:.4f}")
    print(f"  Advanced Reward: {advanced:.4f}")
    assert abs(basic - 1.0) < 0.01, "相同文本应该返回1.0"
    assert abs(advanced - 1.0) < 0.01, "相同文本应该返回1.0"
    print("  ✓ 通过")
    
    # 测试用例2：小幅修改
    original = "Patient takes 10mg aspirin daily."
    revised = "Patient takes 10mg aspirin twice daily."
    
    print(f"\n测试2：小幅修改（添加频率）")
    print(f"  原文: {original}")
    print(f"  修改: {revised}")
    basic = compute_basic_reward(original, revised)
    advanced = compute_advanced_reward(original, revised)
    print(f"  Basic Reward: {basic:.4f}")
    print(f"  Advanced Reward: {advanced:.4f}")
    assert 0.0 <= basic <= 1.0, "Reward应该在[0,1]范围内"
    assert 0.0 <= advanced <= 1.0, "Reward应该在[0,1]范围内"
    print("  ✓ 通过")
    
    # 测试用例3：剂量改变
    original = "Patient takes 10mg aspirin daily."
    revised = "Patient takes 20mg aspirin daily."
    
    print(f"\n测试3：剂量改变（可能更严格的惩罚）")
    print(f"  原文: {original}")
    print(f"  修改: {revised}")
    basic = compute_basic_reward(original, revised)
    advanced = compute_advanced_reward(original, revised)
    print(f"  Basic Reward: {basic:.4f}")
    print(f"  Advanced Reward: {advanced:.4f}")
    assert 0.0 <= basic <= 1.0, "Reward应该在[0,1]范围内"
    assert 0.0 <= advanced <= 1.0, "Reward应该在[0,1]范围内"
    print("  ✓ 通过")
    
    # 测试用例4：空文本
    original = "Patient takes 10mg aspirin daily."
    revised = ""
    
    print(f"\n测试4：删除所有内容（应该很低的reward）")
    print(f"  原文: {original}")
    print(f"  修改: {revised}")
    basic = compute_basic_reward(original, revised)
    advanced = compute_advanced_reward(original, revised)
    print(f"  Basic Reward: {basic:.4f}")
    print(f"  Advanced Reward: {advanced:.4f}")
    assert basic < 0.1, "删除应该返回很低的reward"
    assert advanced < 0.1, "删除应该返回很低的reward"
    print("  ✓ 通过")
    
    # 测试用例5：医学实体变化
    original = "治疗方案：每天服用10mg阿司匹林。"
    revised = "治疗方案：每天服用20mg阿司匹林。"
    
    print(f"\n测试5：医学文本（中文）- 剂量变化")
    print(f"  原文: {original}")
    print(f"  修改: {revised}")
    basic = compute_basic_reward(original, revised)
    advanced = compute_advanced_reward(original, revised)
    print(f"  Basic Reward: {basic:.4f}")
    print(f"  Advanced Reward: {advanced:.4f}")
    assert 0.0 <= basic <= 1.0, "Reward应该在[0,1]范围内"
    assert 0.0 <= advanced <= 1.0, "Reward应该在[0,1]范围内"
    print("  ✓ 通过")
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_reward()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
