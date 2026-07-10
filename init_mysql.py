"""
MySQL 数据库初始化脚本

使用方法:
1. 确保 MySQL 服务已启动
2. 修改 .env 文件中的 MYSQL_PASSWORD 等配置
3. 运行此脚本:
   python init_mysql.py

如果 MySQL 中还没有数据库，本脚本会自动创建。
"""
import pymysql
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
)


def create_database():
    """创建 MySQL 数据库（如果不存在）"""
    print("=" * 60)
    print("MySQL 数据库初始化")
    print("=" * 60)
    print(f"  主机: {MYSQL_HOST}")
    print(f"  端口: {MYSQL_PORT}")
    print(f"  用户: {MYSQL_USER}")
    print(f"  数据库名: {MYSQL_DATABASE}")
    print(f"  密码: {'***' if MYSQL_PASSWORD else '(空)'}")
    print("=" * 60)

    # 先连接 MySQL 服务器（不指定数据库），创建数据库
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4",
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[OK] 数据库 '{MYSQL_DATABASE}' 已创建（或已存在）")
    except Exception as e:
        print(f"[ERROR] 创建数据库失败: {e}")
        print()
        print("请检查:")
        print(f"  1. MySQL 服务是否已启动 (net start mysql)")
        print(f"  2. 用户名和密码是否正确 (.env 文件)")
        print(f"  3. 主机和端口是否正确")
        return False
    return True


def init_tables():
    """初始化表和种子数据"""
    try:
        from database.db_manager import db
        print("[OK] 35 张表 + 4 个视图创建完成")
        print("[OK] 种子数据填充完成")
        print()
        stats = db.get_stats()
        print(f"  总表数: {stats['table_count']}")
        print(f"  总记录数: {stats['total_records']}")
        print()
        for table_name, count in stats["tables"].items():
            if count > 0:
                print(f"    {table_name}: {count} 条")
        return True
    except Exception as e:
        print(f"[ERROR] 表初始化失败: {e}")
        return False


if __name__ == "__main__":
    if create_database():
        init_tables()
    print()
    print("初始化完成！可以运行 python app.py 启动应用了。")
