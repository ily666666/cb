"""
模型工厂 - 根据模型类型名创建模型实例
从 fed/project.py 中的 create_model_by_type 独立出来，
避免导入 fed/project.py 时触发 helper/distiller_zoo 等无关依赖。
"""


def create_model_by_type(model_name, num_classes, dataset_type='ads'):
    """
    根据数据集类型和模型名称创建对应的模型实例

    Args:
        model_name: 模型类型名，如 'complex_resnet50_link11_with_attention'
        num_classes: 分类数
        dataset_type: 数据集类型

    Returns:
        模型实例 (nn.Module)
    """
    if dataset_type == 'radioml':
        if model_name == 'complex_resnet50_radioml':
            from model.complex_resnet50_radioml import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_radioml import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'reii':
        if model_name == 'complex_resnet50_reii':
            from model.complex_resnet50_reii import CombinedModel
            return CombinedModel(num_classes=num_classes)
        else:
            from model.real_resnet20_reii import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'radar':
        if model_name == 'complex_resnet50_radar':
            from model.complex_resnet50_radar import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_radar_with_attention':
            from model.complex_resnet50_radar_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_radar_with_attention_1000':
            from model.complex_resnet50_radar_with_attention_1000 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'real_resnet20_radar_h':
            from model.real_resnet20_radar_h import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.real_resnet20_radar import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'rml2016':
        if model_name == 'complex_resnet50_rml2016':
            from model.complex_resnet50_rml2016 import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_resnet50_rml2016_with_attention':
            from model.complex_resnet50_rml2016_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'real_resnet11_rml2016':
            from model.real_resnet11_rml2016 import ResNet11Real
            return ResNet11Real(num_classes=num_classes)
        elif model_name == 'real_resnet20_rml2016_h':
            from model.real_resnet20_rml2016_h import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.real_resnet20_rml2016 import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    elif dataset_type == 'link11':
        if model_name == 'complex_resnet50_link11':
            from model.complex_resnet50_link11 import ComplexResNet50Link11
            return ComplexResNet50Link11(num_classes=num_classes)
        elif model_name == 'complex_resnet50_link11_with_attention':
            from model.complex_resnet50_link11_with_attention import CombinedModel
            return CombinedModel(num_classes=num_classes)
        elif model_name == 'complex_tiny_link11':
            from model.complex_tiny_link11 import ComplexTinyLink11
            return ComplexTinyLink11(num_classes=num_classes)
        elif model_name == 'real_tiny_link11':
            from model.real_tiny_link11 import RealTinyLink11
            return RealTinyLink11(num_classes=num_classes)
        elif model_name == 'real_resnet9_link11':
            from model.real_resnet9_link11 import ResNet9Real
            return ResNet9Real(num_classes=num_classes)
        elif model_name == 'real_resnet20_link11_h':
            from model.real_resnet20_link11_h import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.real_resnet20_link11 import ResNet20Real
            return ResNet20Real(num_classes=num_classes)

    else:
        if model_name == 'real_resnet20_ads':
            from model.real_resnet20_ads import ResNet20Real
            return ResNet20Real(num_classes=num_classes)
        else:
            from model.complex_resnet50_ads import CombinedModel
            return CombinedModel(num_classes=num_classes)
