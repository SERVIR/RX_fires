/*
//Clip LLP extent to case study area
var llp_cs4 = LLP.clip(CS4)

//Convert LLP extent to vector
var llp4 = llp_cs4.reduceToVectors({
  geometry: CS4,
  eightConnected: true
}).union(1)

//Calculate area in acres(from m2)
var llp_area4 = ee.Number(llp4.geometry().area()).divide(4047)
var percent = llp_area4.divide(1272)
print('CS4 Area (Managed) [acres]', llp_area4, percent)

var cs = ee.FeatureCollection(llp4).style({fillColor:'00000000'})
Map.addLayer(llp4, {}, "LLP CS4")

//Clip LLP extent to case study area
var llp_cs7 = LLP.clip(CS7)

//Convert LLP extent to vector
var llp7 = llp_cs7.reduceToVectors({
  geometry: CS7,
  eightConnected: true
}).union(1)

//Calculate area in acres
var llp_area7 = ee.Number(llp7.geometry().area()).divide(4047)
var percent = llp_area7.divide(2672)
print('CS7 Area(Unmanaged) [acres]', llp_area7, percent)

var cs = ee.FeatureCollection(llp7).style({fillColor:'00000000'})
Map.addLayer(llp7, {}, "LLP CS7")

//Clip LLP extent to case study area
var llp_cs8 = LLP.clip(CS8)

//Convert LLP extent to vector
var llp8 = llp_cs8.reduceToVectors({
  geometry: CS8,
  eightConnected: true
}).union(1)

//Calculate area in acres
var llp_area8 = ee.Number(llp8.geometry().area()).divide(4047)
var percent = llp_area8.divide(1028)
print('CS8 Area (Prescribed) [acres]', llp_area8, percent)

var cs = ee.FeatureCollection(llp8).style({fillColor:'00000000'})
Map.addLayer(llp8, {}, "LLP CS8")

*/


//ONLY NEED TO CHANGE AOI IN THIS SECTION//

//SET DATES: Dates for one year post fire analysis
var startdate = '2019-03-20' //CS7'2019-04-02'//CS4'2019-02-01' //CS8 '2019-03-20'
var enddate = '2020-03-20'//CS7'2020-04-02'//CS4 '2020-02-01'//CS8'2020-03-20'

//SET AOI FOR ANALYSIS
var aoi = CS8

//where 1 acre = 4046.86 m2
var area4 = ee.Number(5107133) //m2 from 1272 acres GIS attribute GISACRES
var area7 =  ee.Number(1.0813e7)//m2 from 2672 acres GIS attribute
var area8 =  ee.Number(4160168)//m2 from 1028 acres GIS attribute

//SET AREA
var area = area8;

print('AOI: CS7, Dates: ', startdate, enddate)
////////////////////////////////////////////////////////////////

//Case Study Fire 4 (1): Discovered 02/01/2019
var cs4 = ee.FeatureCollection(CS4).style({fillColor:'00000000'})
Map.addLayer(cs4, {}, "Case Study 4")

Map.setCenter(-86.79861652832032, 31.03809705947788 , 12);
//Case Study Fire 7 (2): Discovered 04/02/2019
var cs7 = ee.FeatureCollection(CS7).style({fillColor:'00000000'})
Map.addLayer(cs7, {}, "Case Study 7")

//Case Study Fire 8 (3): Discovered 03/20/2019
var cs8 = ee.FeatureCollection(CS8).style({fillColor:'00000000'})
Map.addLayer(cs8, {}, "Case Study 8")

//Merge case study areas
var cs_areas = ee.FeatureCollection([ee.Feature(CS4), ee.Feature(CS7), ee.Feature(CS8)])

//Mask clouds from sentinel 2 image
function maskS2clouds(image) {
  var qa = image.select('QA60');

// Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

// Both flags should be set to zero, indicating clear conditions.

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start', 'system:time_end']);
}

//Sentinel-2 for imagery during the fire
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') 
                  .filterDate(startdate, enddate)
                  .map(maskS2clouds)
                  .filterBounds(aoi)
    
                  
var RGBvis = {
  min: 0.0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2'],
};

Map.addLayer(s2.first(), RGBvis, 'Sentinel Mean') 

//Evapotranspiration
var modis_et = MODIS_ET
  .filterDate(startdate, enddate)
  .filterBounds(AL)
  .select('ET')
  
print(modis_et)

//ET over the following year of the fire
var chart = ui.Chart.image.series({
  imageCollection: modis_et,
  region: aoi,
  reducer:ee.Reducer.mean(),
  scale: 30
}).setOptions({
  interpolateNulls:true,
  lineWidth:1,
  pointSize:3,
  title: "ET TimeSeries for Case Study Area",
  vAxis: {title: 'ET - [kg/m2/8days]'},
  hAxis: {title: 'Date', format: 'YYYY-MMM', gridlines: {count: 12}}
})

print(chart)

//Sum to get annual ET rate
var annual_ET = modis_et.sum()

//Get average annual rate for the study area 1
var annual_ET_CS = annual_ET.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: aoi,
  scale: 500
}) 


var evapotranspirationVis = {
  min: 0,
  max: 300,
  palette:
      ['a50000', 'ff4f1a', 'f1e342', 'c7ef1f', '05fff3', '1707ff', 'd90bff'],
};
  
Map.addLayer(modis_et.mean(), evapotranspirationVis, 'MODIS ET') 

//////////Total precipitation for the year/////////////////////
var prism = PRISM
  .filterDate(startdate, enddate)
  .filterBounds(AL)
  .select('ppt');
  
//Precip over the following year of the fire
var chart = ui.Chart.image.series({
  imageCollection: prism,
  region: aoi,
  reducer:ee.Reducer.mean(),
  scale: 5000
}).setOptions({
  interpolateNulls:true,
  lineWidth:1,
  pointSize:3,
  title: "Precip TimeSeries for Case Study Area",
  vAxis: {title: 'Precipitation [mm]'},
  hAxis: {title: 'Date', format: 'YYYY-MMM', gridlines: {count: 12}}
})

print(chart)

var annual_prism = prism.sum()

//Get average annual precipitation in mm for the study area 1
var annual_prism_CS = annual_prism.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: aoi,
  scale: 5000
})


var precipitationVis = {
  min: 0.0,
  max: 6.0, //mm
  palette: ['red', 'yellow', 'green', 'cyan', 'purple'],
};
    
Map.addLayer(prism.mean(), precipitationVis, 'Precipitation') 

//////////////Calcuate Water Yield////////////////////////////////////
//4047 meters squared = 1 acre
//1kg of water = 0.001 kL
//ET scaling factor = 0.1
//$/kL of water = 0.018
var precip_acre = ee.Number(annual_prism_CS.get('ppt')).multiply(4047).multiply(0.001).multiply(0.018)
var et_acre = ee.Number(annual_ET_CS.get('ET')).multiply(4047).multiply(.01).multiply(0.001).multiply(0.018)
var yield_acre = precip_acre.subtract(et_acre)
print('Mean Annual Precipitation [$/acre]: ', precip_acre)
print('Mean Annual ET [$/acre]: ', et_acre)
print('Water Yield [$/acre]: ', yield_acre)


//Standard Deviation of Water Yield for each CS AOI
var per_pixel_precip_acre = annual_prism.multiply(4047).multiply(0.001).multiply(0.018);
var per_pixel_et_acre = annual_ET.multiply(4047).multiply(0.01).multiply(0.001).multiply(0.018);
var per_pixel_yield = per_pixel_precip_acre.subtract(per_pixel_et_acre)


var stdev = per_pixel_yield.reduceRegion({
  reducer: ee.Reducer.stdDev(),
  geometry: aoi,
  scale: 500,
  bestEffort: true
})

print('Standard Deviation for AOI', stdev)
/////////////////////////////////////////////////////////////////////////////
//Total Value
var precip_cs = ee.Number(annual_prism_CS.get('ppt')).multiply(area)
var et_cs = ee.Number(annual_ET_CS.get('ET')).multiply(area).multiply(.01) //Scaling factor applied 

print('Mean annual ET over CS [kg/year]', et_cs)
print('PRISM average annual precip over the study area', precip_cs)

var water_yield_cs = precip_cs.subtract(et_cs)

print('Annual Post Fire Water Yield for CS [kg]', water_yield_cs)

var water_yield_kL = water_yield_cs.multiply(.0001)

var wy_cost = water_yield_kL.multiply(.018)

print('Water Yield [kL]', water_yield_kL)

print('Water Yield Cost [$]', wy_cost)
  
