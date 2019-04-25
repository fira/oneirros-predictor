require 'influxdb'
require 'csv'
require 'pp'
require 'descriptive_statistics'

CHECKPOINT_END = '1556207526000000000'
CHECKPOINT_START = '1555700000000000000'
TIME_INTERVAL = '1h'
RESOURCES_TYPES = [ 3, 4, 11, 15, 2, 16, 18, 22 ]
BUILDING_TYPES = [ "hospital", "military_base", "school", "missile", "port", "airport", "power_plant", "spaceport", "housing_fund" ]

influxdb = InfluxDB::Client.new 'oneirros', epoch: 's'

def get_statistical_data(records, field)
  a = Array.new
  records.each do |time, record|
    a << record[field].to_f unless record[field].nil?
  end

  return { :min => a.min, :max => a.max, :mean => a.mean, :stddev => a.standard_deviation }
end


def standardize_name(name)
  return name.underscore.tr(' ', '_')
end

def standardize_val(type, coefs, value)
  return value unless not coefs.nil?	# Workaround for non-processable values like time
  return (value - coefs[:mean]) / coefs[:stddev]
end

def normalize_val(type, coefs, value)
  return value unless not coefs.nil?	# Workaround for non-processable values like time
  return (value - coefs[:min]) / (coefs[:max] - coefs[:min])
end

def get_records_r(influxdb, resourceid)
  name = standardize_name(RivalResource.find(resourceid).name)
  query = "SELECT MEAN(lowest_market_price) AS #{name} FROM resource WHERE time < #{CHECKPOINT_END} AND time > #{CHECKPOINT_START} AND rivals_resource_id = '#{resourceid}' GROUP BY time(#{TIME_INTERVAL}) FILL(none) ORDER BY time DESC"
  result = influxdb.query query
  return result.first["values"]
end

def get_records_b(influxdb) 
  btypes = Array.new
  for b in BUILDING_TYPES
    btypes << "MEAN(#{b}) AS #{b}"
  end

  query = "SELECT #{btypes.join(',')} FROM region WHERE time < #{CHECKPOINT_END} AND time > #{CHECKPOINT_START} GROUP BY time(#{TIME_INTERVAL}) FILL(none) ORDER BY time DESC"
  result = influxdb.query query
  return result.first["values"]
end

def merge_records(merged_records, records)
  for record in records
     if merged_records[record["time"]].nil?
       merged_records[record["time"]] = Hash.new
       merged_records[record["time"]]["time"] = record["time"]
     end
     merged_records[record["time"]].merge!(record.except("time"))
  end
  return merged_records
end

allcoefs = Hash.new
merged_records = Hash.new

for r in RESOURCES_TYPES
  puts "Getting and merging Records for Resource = #{r}"
  r_records = get_records_r(influxdb, r)
  merge_records(merged_records, r_records)
end

puts "Getting Building Records"
b_records = get_records_b(influxdb)
puts "Merging Building Records"
merge_records(merged_records, b_records)

CSV.open('norm_coeffs.csv', 'w') do |csv|
  # Write header
  csv << [ "variable", "min", "max", "mean", "stddev" ]

  # Write coeffs
  for r in RESOURCES_TYPES
    name = standardize_name(RivalResource.find(r).name)
    coefs = get_statistical_data(merged_records, name)
    csv << [ name, coefs[:min], coefs[:max], coefs[:mean], coefs[:stddev] ]
    allcoefs[name] = coefs
    pp coefs
  end

  # Write coeffs
  for b in BUILDING_TYPES
    coefs = get_statistical_data(merged_records, b)
    csv << [ b, coefs[:min], coefs[:max], coefs[:mean], coefs[:stddev] ]
    allcoefs[b] = coefs
    pp coefs
  end

  puts "Done writing Coefficients CSV"
end

header = Array.new
header << "time"
for b in BUILDING_TYPES
  header << b
end

for r in RESOURCES_TYPES
  header << standardize_name(RivalResource.find(r).name)
end

CSV.open('norm_data.csv', 'w') do |csv|
  csv << header 

  puts "Starting to write normalized values CSV"
  merged_records.each do |time, record|
    row = Array.new
    for field in header
      row << normalize_val(field, allcoefs[field], record[field].to_f)
    end
    csv << row
  end

  puts "Done."

end

CSV.open('std_data.csv', 'w') do |csv|
  csv << header

  puts "Starting to write standardized values CSV"
  merged_records.each do |time, record|
    row = Array.new
    for field in header
      row << standardize_val(field, allcoefs[field], record[field].to_f)
    end
    csv << row
  end

  puts "Done."

end

puts "All Done. Good day to you !"
