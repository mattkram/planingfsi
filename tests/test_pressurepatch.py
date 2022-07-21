# from __future__ import annotations
#
# import pytest
#
# from planingfsi.fsi.simulation import Simulation
# from planingfsi.potentialflow.pressurepatch import PlaningSurface
# from planingfsi.potentialflow.pressurepatch import PressureCushion
# from planingfsi.potentialflow.pressurepatch import PressurePatch
# from planingfsi.potentialflow.solver import PotentialPlaningSolver
#
# TEST_NAME = "test_name"
#
#
# def test_must_set_parent_to_access_config() -> None:
#     surface = PlaningSurface()
#     with pytest.raises(ValueError):
#         surface.config
#
#
# # TODO: This parent= thing has to go
#
#
# @pytest.mark.parametrize(
#     "named_pressure_patch",
#     [
#         PlaningSurface({"substructureName": TEST_NAME}),
#         PlaningSurface({"substructureName": "wrong_name"}, name=TEST_NAME),
#         PressureCushion(
#             {"pressureCushionName": TEST_NAME, "cushionPressure": 0.0},
#             parent=PotentialPlaningSolver(Simulation()),
#         ),
#         PressureCushion(name=TEST_NAME, parent=PotentialPlaningSolver(Simulation())),
#     ],
# )
# def test_pressure_patch_name(named_pressure_patch: PressurePatch) -> None:
#     assert named_pressure_patch.name == TEST_NAME
