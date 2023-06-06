from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('stock_scanner_app', '0001_initial'),  # Replace '0001_initial' with the actual initial migration file name of the stock_scanner_app if it's different
    ]

    operations = [
        migrations.CreateModel(
            name='TrainingData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=20)),
                ('date', models.DateField()),
                ('open', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('close', models.FloatField()),
                ('volume', models.FloatField()),
                ('company', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='stock_scanner_app.CompanyInfo')),
            ],
        ),
    ]
