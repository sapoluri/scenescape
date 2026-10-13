# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.db import migrations, models


class Migration(migrations.Migration):

  dependencies = [
    ("manager", "0008_asset3d_scale_xyz"),
  ]

  operations = [
    migrations.AddField(
      model_name="asset3d",
      name="mark_geometry",
      field=models.CharField(
        choices=[
          ("box", "Box"),
          ("capsule", "Capsule"),
          ("cylinder", "Cylinder"),
        ],
        default="box",
        help_text="Box, capsule, or cylinder used when no GLB is set",
        max_length=16,
        verbose_name="Mark geometry",
      ),
    ),
  ]
