using LlmsFromScratch.DotNet.Shared.Tensors;

namespace LlmsFromScratch.DotNet.Shared.Nn;

/// <summary>
/// 神经网络模块基类（对应 PyTorch 的 nn.Module）
/// 所有层和模型都继承此类
/// 提供参数收集、训练/推理模式切换、梯度清零等功能
/// </summary>
public abstract class Module
{
    /// <summary>是否处于训练模式（影响 Dropout 等行为）</summary>
    public bool IsTraining { get; private set; } = true;

    /// <summary>子模块字典，用于参数发现</summary>
    protected Dictionary<string, Module> SubModules { get; } = new();

    /// <summary>参数字典</summary>
    protected Dictionary<string, Tensor> Params { get; } = new();

    /// <summary>前向传播（子类必须实现）</summary>
    public abstract Tensor Forward(Tensor input);

    /// <summary>注册子模块</summary>
    protected void RegisterModule(string name, Module module)
    {
        SubModules[name] = module;
    }

    /// <summary>注册可训练参数</summary>
    protected void RegisterParameter(string name, Tensor param)
    {
        param.RequiresGrad = true;
        Params[name] = param;
    }

    /// <summary>递归获取所有可训练参数</summary>
    public virtual IEnumerable<Tensor> Parameters()
    {
        foreach (var param in Params.Values)
            yield return param;

        foreach (var module in SubModules.Values)
            foreach (var param in module.Parameters())
                yield return param;
    }

    /// <summary>递归获取所有命名子模块</summary>
    public virtual IEnumerable<(string Name, Module Module)> NamedModules(string prefix = "")
    {
        foreach (var (name, module) in SubModules)
        {
            string fullName = prefix + name;
            yield return (fullName, module);
            foreach (var child in module.NamedModules(fullName + "."))
                yield return child;
        }
    }

    /// <summary>
    /// 替换指定名称的子模块
    /// </summary>
    public void ReplaceSubModule(string name, Module newModule)
    {
        SubModules[name] = newModule;
    }

    /// <summary>递归获取所有命名参数</summary>
    public virtual IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        foreach (var (name, param) in Params)
            yield return (prefix + name, param);

        foreach (var (modName, module) in SubModules)
            foreach (var np in module.NamedParameters(prefix + modName + "."))
                yield return np;
    }

    /// <summary>递归设置训练/推理模式</summary>
    public void SetTraining(bool training)
    {
        IsTraining = training;
        foreach (var module in SubModules.Values)
            module.SetTraining(training);
    }

    /// <summary>清零所有参数的梯度</summary>
    public void ZeroGrad()
    {
        foreach (var param in Parameters())
            param.ZeroGrad();
    }
}
