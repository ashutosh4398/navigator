# Generated by Django 3.0.5 on 2020-05-15 15:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('atgexp', '0003_auto_20200515_2029'),
    ]

    operations = [
        migrations.RenameField(
            model_name='avatar',
            old_name='image',
            new_name='image_url',
        ),
    ]
