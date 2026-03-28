# tests/test_tax_report.py
from datetime import date
from taxopt.data_types import TaxLot
from taxopt.tax_report import RealizedGain, TaxReport
from taxopt.tax_policy import USCapitalGainsPolicy

POLICY = USCapitalGainsPolicy()

def _gain(gain_type: str, gain: float) -> RealizedGain:
    lot = TaxLot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    return RealizedGain(
        asset="AAPL", lot=lot, quantity=10.0,
        proceeds=150.0, cost=100.0, gain=gain, gain_type=gain_type,
    )


def test_from_events_aggregates_by_type():
    events = [_gain("long_term", 200.0), _gain("long_term", 100.0), _gain("short_term", 50.0)]
    report = TaxReport.from_events(events, POLICY, date(2024, 1, 1))
    assert report.totals_by_type["long_term"] == 300.0
    assert report.totals_by_type["short_term"] == 50.0


def test_from_events_total_gain():
    events = [_gain("long_term", 300.0), _gain("short_term", 50.0)]
    report = TaxReport.from_events(events, POLICY, date(2024, 1, 1))
    assert report.total_gain == 350.0


def test_from_events_empty():
    report = TaxReport.from_events([], POLICY, date(2024, 1, 1))
    assert report.total_gain == 0.0
    assert report.events == []


def test_tax_liability():
    events = [_gain("long_term", 1000.0), _gain("short_term", 500.0)]
    report = TaxReport.from_events(events, POLICY, date(2024, 1, 1))
    expected = 1000.0 * POLICY.lt_rate + 500.0 * POLICY.st_rate
    assert abs(POLICY.tax_liability(report) - expected) < 1e-9


def test_netting_applied_in_report():
    # ST loss should offset LT gain before total is computed
    events = [_gain("long_term", 500.0), _gain("short_term", -200.0)]
    report = TaxReport.from_events(events, POLICY, date(2024, 1, 1))
    assert report.totals_by_type["long_term"] == 300.0
    assert report.totals_by_type["short_term"] == 0.0
