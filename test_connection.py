"""Quick test to verify Supabase connection"""
import os
import sys

# Check environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Supabase environment variables not set")
    print("\nSet them in PowerShell:")
    print('  $env:SUPABASE_URL = "your-project-url"')
    print('  $env:SUPABASE_KEY = "your-service-role-key"')
    sys.exit(1)

print("✓ Environment variables found")

# Test connection
try:
    from supabase import create_client
    print("✓ Supabase library imported")
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Client created")
    
    # Test users table
    result = supabase.table("users").select("user_id").limit(1).execute()
    print(f"✓ 'users' table: OK ({len(result.data)} rows)")
    
    # Test activity_logs table
    result = supabase.table("activity_logs").select("id").limit(1).execute()
    print(f"✓ 'activity_logs' table: OK ({len(result.data)} rows)")
    
    print("\n✅ All tests passed! Database is ready.")
    
except ImportError:
    print("❌ Supabase library not installed")
    print("   Run: pip install supabase")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n💡 Make sure you created the tables in Supabase SQL Editor")
