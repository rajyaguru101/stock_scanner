from django.contrib import admin
from .models import CompanyInfo, ScannerConfiguration, HistoricalData

class CompanyInfoAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'company_name', 'exchange', 'sector', 'country')
    search_fields = ('symbol', 'company_name',)
    list_filter = ('exchange', 'sector', 'country')

class ScannerConfigurationAdmin(admin.ModelAdmin):
    list_display = ('name', 'scanner_type')
    search_fields = ('name',)
    list_filter = ('scanner_type',)

class HistoricalDataAdmin(admin.ModelAdmin):
    list_display = ('company', 'date', 'open', 'high', 'low', 'close', 'volume')
    search_fields = ('company__symbol', 'company__company_name',)
    list_filter = ('company__exchange', 'company__sector', 'company__country', 'date')


admin.site.register(CompanyInfo, CompanyInfoAdmin)
admin.site.register(ScannerConfiguration, ScannerConfigurationAdmin)
admin.site.register(HistoricalData, HistoricalDataAdmin)


