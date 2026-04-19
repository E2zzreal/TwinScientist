# tests/interface/test_trigger_detector.py
import pytest
from unittest.mock import MagicMock
from interface.trigger_detector import TriggerDetector, TriggerType


def _make_detector(twin_name="张三", domains=None):
    return TriggerDetector(
        twin_name=twin_name,
        confident_domains=domains or ["氢能催化剂", "电解水", "材料表征"],
    )


def test_detect_direct_mention():
    """Should trigger when twin is directly named."""
    detector = _make_detector(twin_name="张三")
    result = detector.check("李四：张三老师你怎么看这个数据？")
    assert result.should_speak is True
    assert result.trigger_type == TriggerType.DIRECT_MENTION
    assert result.urgency == "high"


def test_detect_domain_relevance():
    """Should suggest speaking when own domain is discussed."""
    detector = _make_detector(domains=["氢能催化剂", "HER"])
    result = detector.check("李四：最近HER催化剂的研究进展怎么样？")
    assert result.should_speak is True
    assert result.trigger_type == TriggerType.DOMAIN_RELEVANT


def test_no_trigger_for_irrelevant_content():
    """Should not trigger for off-domain content."""
    detector = _make_detector(domains=["氢能催化剂"])
    result = detector.check("李四：大家今天的午饭吃什么？")
    assert result.should_speak is False


def test_detect_question_in_domain():
    """Direct question about twin's domain should trigger."""
    detector = _make_detector(domains=["催化剂稳定性", "单原子催化"])
    result = detector.check("有没有人研究过单原子催化在长期稳定性方面的进展？")
    assert result.should_speak is True


def test_trigger_urgency_levels():
    """Direct mention should be higher urgency than domain match."""
    detector = _make_detector(twin_name="张三", domains=["氢能"])
    direct = detector.check("张三你来说说？")
    domain = detector.check("氢能方面有什么新进展？")
    assert direct.urgency == "high"
    assert domain.urgency in ("medium", "low")