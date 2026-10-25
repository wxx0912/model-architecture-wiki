# 来源层

`INDEX.generated.md` 是按主题分组、含引用计数的生成索引，不可手改。

来源页回答“读到了什么、这份来源不能支持什么”，正文页承载结构与实现的解释。一来源一文件：论文在 `papers/`，代码库在 `repos/`，官方文档与知识库在 `docs/`。主题级说明与边界见 `topics/`。

```bash
python3 scripts/get_page.py component-attention --follow-sources   # 读页面并展开其来源页
python3 scripts/validate.py                                        # 校验来源字段、正文结构与引用
python3 scripts/build_index.py                                     # 生成 index.generated.md 与 INDEX.generated.md
```

来源存在不意味着整篇所有结论都已验证：读取形式、固定版本、已读定位和不能推出的结论都记在各自的来源页里。本库不镜像论文全文或上游代码。
