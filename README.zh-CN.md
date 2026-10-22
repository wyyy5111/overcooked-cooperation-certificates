# Overcooked Cooperation Certificates

本项目提供厨房协作产出认证方法的核心实现：原生事件流建模、精确有理数证书核验、有限时长产出上界、可重复生产调度，以及独立局部时钟的组合。

## 安装与运行

使用Python 3.10或更新版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[solver,native]'
occ demo --layout ring --out results/ring
```

Windows环境使用`.venv\Scripts\activate`激活虚拟环境。完整命令与接口说明见[英文README](README.md)。

该示例依次构建两个固定角色的单人模型、验证精确上界、执行并核验双人生产序列，最后比较持续产出与有限时长产出。Ring的单人产出率上界为`1/42`，可实现双人产出率为`3/97`，资源增益保证为`29/97`。

## 核心入口

- `occ solve`：求解事件流模型，并以精确有理数核验所得证书。
- `occ check`：独立检查已保存模型的列结构和证书运算。
- `occ schedule`：生成并确认原生执行中的可重复生产序列。
- `occ verify-schedule`：重新执行保存的动作，核验周期状态及产出。
- `occ finite`：计算空初始状态下的有限产出上界，或读取/外推已核验序列的产出。
- `occ certify`：将单人上界与双人可实现产出进行比较。

自动调度在满足区域和资源独立条件的布局上使用异步局部时钟，其余布局保留同步执行。输出中明确区分实测产出、周期外推产出和通过局部周期构造的联合周期。

```bash
python -m unittest discover -s tests -v
```

代码采用MIT许可证。原生引擎与布局保留Overcooked-AI的原始MIT声明及固定版本来源。
