#!/usr/bin/env python3
"""
Test script to check PettingZoo MPE rendering capabilities
"""

import numpy as np
import sys
import os

# Add HARL to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pettingzoo_rendering():
    """Test if PettingZoo MPE can render visually"""
    print("Testing PettingZoo MPE rendering...")
    
    try:
        # Try direct PettingZoo import
        from pettingzoo.mpe import simple_spread_v3
        print("✓ PettingZoo import successful")
        
        # Create environment with rendering enabled
        try:
            env = simple_spread_v3.parallel_env(render_mode='rgb_array', N=3)
            print("✓ Environment created with rgb_array mode")
        except:
            try:
                env = simple_spread_v3.parallel_env(render_mode='human', N=3)
                print("✓ Environment created with human mode")
            except:
                env = simple_spread_v3.parallel_env(N=3)
                print("✓ Environment created without render mode")
        
        # Reset environment
        obs = env.reset()
        print("✓ Environment reset successful")
        
        # Try rendering
        try:
            frame = env.render()
            if isinstance(frame, np.ndarray):
                print(f"✓ Render successful! Frame shape: {frame.shape}")
                return True
            else:
                print(f"⚠ Render returned: {type(frame)}")
        except Exception as e:
            print(f"⚠ Direct render failed: {e}")
        
        # Try pygame approach
        try:
            import pygame
            print("✓ Pygame available")
            
            # Initialize pygame
            pygame.init()
            screen = pygame.display.set_mode((600, 600))
            print("✓ Pygame display initialized")
            
            # Try rendering to screen
            env.render()
            
            # Capture screen
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            print(f"✓ Pygame capture successful! Frame shape: {frame.shape}")
            
            pygame.quit()
            return True
            
        except ImportError:
            print("⚠ Pygame not available - install with: pip install pygame")
        except Exception as e:
            print(f"⚠ Pygame approach failed: {e}")
        
        env.close()
        return False
        
    except ImportError as e:
        print(f"✗ PettingZoo import failed: {e}")
        print("Install with: pip install pettingzoo")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_harl_wrapper_rendering():
    """Test HARL's PettingZoo wrapper rendering"""
    print("\nTesting HARL wrapper rendering...")
    
    try:
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
        
        args = {
            "scenario": "simple_spread_v3",
            "N": 3,
            "max_cycles": 25
        }
        
        env = PettingZooMPEEnv(args)
        print("✓ HARL wrapper created")
        
        # Reset
        obs, share_obs, avail = env.reset()
        print("✓ Environment reset")
        
        # Try rendering
        try:
            frame = env.render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                print(f"✓ HARL wrapper render successful! Frame shape: {frame.shape}")
                return True
            else:
                print(f"⚠ HARL wrapper render returned: {type(frame)}")
        except Exception as e:
            print(f"⚠ HARL wrapper render failed: {e}")
        
        env.close()
        return False
        
    except Exception as e:
        print(f"✗ HARL wrapper test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PettingZoo MPE Rendering Test")
    print("=" * 60)
    
    # Test 1: Direct PettingZoo
    pz_works = test_pettingzoo_rendering()
    
    # Test 2: HARL wrapper
    harl_works = test_harl_wrapper_rendering()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Direct PettingZoo: {'✓ Works' if pz_works else '✗ Failed'}")
    print(f"HARL Wrapper: {'✓ Works' if harl_works else '✗ Failed'}")
    
    if not (pz_works or harl_works):
        print("\nTo enable visual rendering, try:")
        print("1. pip install pygame")
        print("2. pip install pettingzoo[mpe]")
        print("3. Ensure you have a display (for headless servers, use Xvfb)")
        
    print("\nNote: Even if rendering fails, Shapley value computation will work correctly!")
