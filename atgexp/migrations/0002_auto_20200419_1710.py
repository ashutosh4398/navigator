# Generated by Django 3.0.3 on 2020-04-19 11:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('atgexp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='avatar',
            name='image',
            field=models.ImageField(default='blank', upload_to='images'),
            preserve_default=False,
        ),
    ]