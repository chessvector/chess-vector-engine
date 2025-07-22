#!/bin/bash
echo "🎯 Testing Chess Vector Engine..."
echo "Sending UCI commands to verify engine functionality..."

echo -e "uci\nposition startpos\ngo movetime 1000\nquit" | ./chess-vector-engine

echo ""
echo "✅ Test complete! If you see 'bestmove' output above, the engine is working correctly."
echo "📚 See INSTALL.md for chess GUI setup instructions."
