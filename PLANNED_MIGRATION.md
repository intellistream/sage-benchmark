# ⚠️ Benchmark 即将迁移到独立仓库

**sage-benchmark 即将迁移到独立仓库**: https://github.com/intellistream/sage-benchmark（新创建）

**注意**: 这是 SAGE 框架特定的 benchmark，不要与 OmniBenchmark（组织级综合benchmark集合）混淆。

## 当前状态

本目录 (`SAGE/benchmark/`) 包含 SAGE 系统级 benchmark 和评估框架。

**计划迁移**: 本目录将迁移到独立的 `sage-benchmark` 仓库，并发布为 PyPI 包 `isage-benchmark`。

## 迁移进度

- [x] 迁移计划文档 (`MIGRATION_TO_INDEPENDENT_REPO.md`)
- [ ] 创建独立仓库
- [ ] 重构包结构
- [ ] 更新导入路径
- [ ] 设置 CI/CD
- [ ] 发布 PyPI 包
- [ ] 更新 SAGE 主仓库文档
- [ ] 清理主仓库 benchmark 目录

## 迁移完成后

迁移完成后，请使用：

```bash
pip install isage-benchmark
```

## 详细信息

查看完整迁移指南: [MIGRATION_TO_INDEPENDENT_REPO.md](./MIGRATION_TO_INDEPENDENT_REPO.md)

## 相关链接

- **独立仓库** (即将创建): https://github.com/intellistream/sage-benchmark
- **PyPI 包** (即将发布): https://pypi.org/project/isage-benchmark/
- **迁移文档**: [MIGRATION_TO_INDEPENDENT_REPO.md](./MIGRATION_TO_INDEPENDENT_REPO.md)
