# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from django.db import migrations, models


def copy_uniform_scale(apps, schema_editor):
  Asset3D = apps.get_model("manager", "Asset3D")
  for asset in Asset3D.objects.all().iterator():
    s = asset.scale if asset.scale is not None else 1.0
    Asset3D.objects.filter(pk=asset.pk).update(
      scale_x=s, scale_y=s, scale_z=s,
    )


class Migration(migrations.Migration):

  dependencies = [
    ("manager", "0007_meshgenerationrequest"),
  ]

  operations = [
    migrations.AddField(
      model_name="asset3d",
      name="scale_x",
      field=models.FloatField(
        blank=True, default=1.0, null=True, verbose_name="X Scale",
      ),
    ),
    migrations.AddField(
      model_name="asset3d",
      name="scale_y",
      field=models.FloatField(
        blank=True, default=1.0, null=True, verbose_name="Y Scale",
      ),
    ),
    migrations.AddField(
      model_name="asset3d",
      name="scale_z",
      field=models.FloatField(
        blank=True, default=1.0, null=True, verbose_name="Z Scale",
      ),
    ),
    migrations.RunPython(copy_uniform_scale, migrations.RunPython.noop),
  ]
