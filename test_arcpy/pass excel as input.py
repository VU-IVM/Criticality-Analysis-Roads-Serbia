import arcpy

excel_path = arcpy.GetParameterAsText(0)

arcpy.env.workspace = excel_path
sheets = arcpy.ListTables()

arcpy.AddMessage(f"Sheets found: {sheets}")

sheet = sheets[0]
