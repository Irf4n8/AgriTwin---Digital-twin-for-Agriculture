window.commonMappingData = {};
window.onlyVegetableValue = "";
window.currentStartDate = "";
window.currentEndDate = "";
window.flatpickrInstance = null;
var pageType = document.documentElement.dataset.pageType;
window.selectedVegetablesCache = "selectedVegetablesCache_"+pageType;
window.historyPageApiPathChartDataId = "chartdatavalues";
window.apiPathMarketPagesId = "dataapi";

function waitForLibraries(callback, retryCount) {
	retryCount = retryCount || 0;
	if(typeof jQuery == 'function' && typeof getFromLocalStorage === 'function') {
		callback();
	} else if(retryCount < 200) {
		setTimeout(function(){
			waitForLibraries(callback, retryCount + 1);
		}, 100);
	}
}

waitForLibraries(function(){
	$(function(){
	    initHistoryPage();
	});
});

function initHistoryPage()
{
	var searchParams = new URLSearchParams(window.location.search);
    if(searchParams.has("onlyVegetable"))
    {
        window.onlyVegetableValue = searchParams.get("onlyVegetable");
    }
	var days = getNumberOfDays();
	setTimeout(function(){
		updateLatestRangeData(days);
	}, 10);
}

function getItemName()
{
	var pagetype = document.documentElement.dataset.pageType;
	var names = {"market":"vegetable", "fruits":"fruits", "nonveg":"nonveg"};
	var name = names[pagetype];
	if(name)
	{
		return name;
	}
	else
	{
		return "item";
	}
}

function getNumberOfDays()
{
    var pageWidth = window.innerWidth;
    if(pageWidth > 1200)
    {
        return 30;
    }
    else if(pageWidth > 768)
    {
        return 15;
    }
    else
    {
        return 7;
    }
}

function loadDatePicker(startDate, endDate, startDateLimit, retryCount)
{
    retryCount = retryCount || 0;
    
    if(typeof flatpickr == "undefined" || typeof jQuery != 'function')
    {
        if(retryCount < 20)
        {
            setTimeout(function(){
                loadDatePicker(startDate, endDate, startDateLimit, retryCount + 1);
            }, 1000);
        }
        return;
    }
    
    var start = new Date(startDate);
    var end = new Date(endDate);
	window.currentStartDate = start;
	window.currentEndDate = end;

    function cb(selectedDates) {
        if(selectedDates.length === 2) {
            window.currentStartDate = selectedDates[0];
            window.currentEndDate = selectedDates[1];
            updateSpan(selectedDates[0], selectedDates[1]);
            rangeSelectionUpdatedHistory();
            updateRangeNavigationButtons();
        }
    }

	function updateSpan(start, end)
	{
		var formatted;
		if(window.innerWidth > 576)
		{
			formatted = formatDate(start, 'MMMM D, YYYY') + ' - ' + formatDate(end, 'MMMM D, YYYY');
		}
		else
		{
			formatted = formatDate(start, 'MMM D, YYYY') + ' - ' + formatDate(end, 'MMM D, YYYY');
		}
		$('#reportrange span').html(formatted);
	}

    window.flatpickrInstance = flatpickr('#reportrange', {
        mode: 'range',
        dateFormat: 'Y-m-d',
        defaultDate: [start, end],
        minDate: startDateLimit,
        maxDate: end,
        onChange: cb
    });
	updateSpan(start, end);
	updateRangeNavigationButtons();
	
	document.getElementById('prevRangeBtn').onclick = function() { changeRangeByOffset(-1); };
	document.getElementById('nextRangeBtn').onclick = function() { changeRangeByOffset(1); };
	document.getElementById('navItemTable').onclick = function() { changeHistoryPageView('table'); };
	document.getElementById('navItemChart').onclick = function() { changeHistoryPageView('chart'); };
	document.querySelector('.history-data-chart-table-container').onclick = clickedInsideChart;
}

function updateRangeNavigationButtons() {
    if(!window.flatpickrInstance || !window.flatpickrInstance.selectedDates.length) return;
    
    var dates = window.flatpickrInstance.selectedDates;
    var startDate = dates[0];
    var endDate = dates[1] || dates[0];
    var minDate = new Date(window.flatpickrInstance.config.minDate);
    var maxDate = new Date(window.flatpickrInstance.config.maxDate);
    
    document.getElementById('prevRangeBtn').style.display = 
        startDate <= minDate ? 'none' : 'inline-block';
    document.getElementById('nextRangeBtn').style.display = 
        endDate >= maxDate ? 'none' : 'inline-block';
}

function changeRangeByOffset(direction) {
    if(!window.flatpickrInstance || !window.flatpickrInstance.selectedDates.length) return;
    
    var dates = window.flatpickrInstance.selectedDates;
    var startDate = new Date(dates[0]);
    var endDate = new Date(dates[1] || dates[0]);
    var daysDiff = Math.round((endDate - startDate) / (1000 * 60 * 60 * 24));
    
    var newStart = new Date(startDate);
    var newEnd = new Date(endDate);
    newStart.setDate(newStart.getDate() + (direction * (daysDiff + 1)));
    newEnd.setDate(newEnd.getDate() + (direction * (daysDiff + 1)));
    
    var minDate = new Date(window.flatpickrInstance.config.minDate);
    var maxDate = new Date(window.flatpickrInstance.config.maxDate);
    
    if(newStart >= minDate && newEnd <= maxDate) {
        window.flatpickrInstance.setDate([newStart, newEnd], true);
    }
    updateRangeNavigationButtons();
}

function formatDate(date, format) {
    var d = new Date(date);
    var months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    if (format === 'MMMM D, YYYY') {
        return months[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear();
    } else if (format === 'MMM D, YYYY') {
        return months[d.getMonth()].substr(0, 3) + ' ' + d.getDate() + ', ' + d.getFullYear();
    } else if (format === 'YYYY-MM-DD') {
        return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
    } else if (format === 'MMM DD') {
        return months[d.getMonth()].substr(0, 3) + ' ' + String(d.getDate()).padStart(2, '0');
    }
    return d.toISOString().split('T')[0];
}

function updateLatestRangeData(days)
{
	var marketName = document.getElementById("mainContainer").dataset.marketName;
	getLatestChartData(marketName, days, function(data){
		try
		{
		    var startDate, startDateLimit, endDate;
		    if(data)
		    {
				startDate = data.startDate;
				startDateLimit = data.startDateLimit;
				endDate = data.endDate;
			    loadDatePicker(startDate, endDate, startDateLimit);
				updateHistoryData(data, startDate, endDate);
				updateMultiSelect(data.columnMapping);
			}
			else
			{
				var today = new Date();
				startDate = formatDate(today, 'YYYY-MM-DD');
				var fiveYearsAgo = new Date();
				fiveYearsAgo.setFullYear(fiveYearsAgo.getFullYear() - 5);
				startDateLimit = formatDate(fiveYearsAgo, 'YYYY-MM-DD');
				endDate = formatDate(today, 'YYYY-MM-DD');
			    loadDatePicker(startDate, endDate, startDateLimit);
			    $("#history-chart-container").html("No data found");
			}
		}
		catch(e)
		{
			console.log("Clearing localstorage cause failure {}", e);
			setFromLocalStorage(window.selectedVegetablesCache, "");
		}
	}, function(data)
	{
		var today = new Date();
		startDate = formatDate(today, 'YYYY-MM-DD');
		var fiveYearsAgo = new Date();
		fiveYearsAgo.setFullYear(fiveYearsAgo.getFullYear() - 5);
		startDateLimit = formatDate(fiveYearsAgo, 'YYYY-MM-DD');
		endDate = formatDate(today, 'YYYY-MM-DD');
	    loadDatePicker(startDate, endDate, startDateLimit);
	    if(data && data.errorMessage)
	    {
			$("#history-chart-container").html(data.errorMessage);
		}
		else
		{
			$("#history-chart-container").html("No data found");
		}
	});
}

function rangeSelectionUpdatedHistory()
{
	var start = window.currentStartDate;
	var end = window.currentEndDate;
	var marketName = document.getElementById("mainContainer").dataset.marketName;
    var selectedVegetables = getSelectValues();
	getChartDataFromUrl(marketName, start, end, selectedVegetables, function(data){
	    var startDate = data.startDate;
	    var endDate = data.endDate;
		updateHistoryData(data, startDate, endDate);
	});
}

var multipleButton;
function updateMultiSelect(columnMapping)
{
	var optionsData = "";
	for(var k=0;k<columnMapping.length;k++)
    {
		var selected = "";
		if(columnMapping[k]["selected"] == true)
		{
			selected = "selected"
		}
        optionsData = optionsData + "<option value='" + columnMapping[k]["vegId"] + "' "+
				selected+">" + columnMapping[k]["vegName"] + "</option>";
    }
    $("#choices-multiple-button").html(optionsData);
    multipleButton = new Choices('#choices-multiple-button', {
        removeItemButton: true,
        maxItemCount:8,
        searchResultLimit:3,
        renderChoiceLimit:3,
		placeholderValue: 'Select ' + getItemName(),
    });
    // $("#choices-multiple-button-container input").attr("id","selectVetableInput");
    // $("#choices-multiple-button-container input").after("<label for=\"selectVetableInput\" style=\"cursor:text;opacity: 0.5;\">Click to here add</label>");
}

function clickedInsideChart()
{
	$("#choices-multiple-button-container input").blur();
}

function selectVegetableChange()
{
	rangeSelectionUpdatedHistory();
	var selectedValues = getSelectValues();
	setFromLocalStorage(window.selectedVegetablesCache, selectedValues);
	if(window.location.search && window.location.search != "")
	{
		window.history.replaceState(null, null, window.location.pathname);
	}
}

function getSelectValues() {
  var select = $("#choices-multiple-button")[0];
  var result = [];
  var options = select && select.options;
  var opt;

  for (var i=0, iLen=options.length; i<iLen; i++) {
    opt = options[i];
    if (opt.selected) {
      result.push(opt.value || opt.text);
    }
  }
  return result;
}

function updateHistoryData(content, startDate, endDate)
{
    var fromTo = startDate + " to " + endDate;
	var chartTitle = "History data of " + fromTo;
    var chartDescription = document.getElementById("pageContentHeaderTitle").innerText + " of " + fromTo;
	loadDataInTable(content);
    loadDataInChart(content, chartTitle, chartDescription);	
}

function getLatestChartData(marketName, days, callback, failureCallback)
{
	var vegId = "";
	if(window.onlyVegetableValue)
	{
		vegId = "&vegIds=" + window.onlyVegetableValue;
	}
	else
	{
		var selectedvalue = getFromLocalStorage(window.selectedVegetablesCache);
		if(selectedvalue && selectedvalue != "")
		{
			vegId = "&vegIds=" + selectedvalue;
		}
	}
	var pageType = document.getElementById("mainContainer").dataset.pageType;

    var url = "/api/"+window.apiPathMarketPagesId+"/"+pageType+"/" + marketName + "/latestchartdata?days=" + days + vegId;
	callAjax(url, function(data){
		callback(data);
	}, function(data){
		setFromLocalStorage(window.selectedVegetablesCache, "");
		failureCallback(data);
	});
}

function getChartDataFromUrl(marketName, start, end, selectedVegetables, callback)
{
    var startDate = formatDate(start, 'YYYY-MM-DD');
    var endDate = formatDate(end, 'YYYY-MM-DD');
	var vegIds = selectedVegetables.join(','); 
	var pageType = document.getElementById("mainContainer").dataset.pageType;
    var url = "/api/"+window.apiPathMarketPagesId+"/"+pageType+"/" + marketName + "/"+window.historyPageApiPathChartDataId+"?start=" + startDate + "&end=" + endDate + "&vegIds="+vegIds;
    callAjax(url, function(market_data){
        window.cachedMarketData[url] = market_data;
        callback(market_data);
	});
}

function loadDataInTable(content)
{
    $("#historyDataTable").html("");
    var columns = content.columns;
    var data = content.data;
    var headerRow = "<thead><tr><th></th>";
    for(var i=0;i<columns.length;i++)
    {
        var col = columns[i];
        headerRow = headerRow + "<th><span style=\"white-space: nowrap;\">" + col + "</span></th>";
    }
    headerRow = headerRow + "</tr></thead>";

    var tbodyRows = "<tbody>";
    var dataKeys = Object.keys(data);
    for(var i=0;i<dataKeys.length;i++)
    {
        var dataKey = dataKeys[i];
        var dataMap = data[dataKey];
        tbodyRows = tbodyRows + "<tr><td>" + dataMap["name"] + "</td>";
        var dataValuesList = dataMap["data"];
        if(dataValuesList)
		{
	        for(var j=0;j<dataValuesList.length;j++)
	        {
	            var value = dataValuesList[j];
	            var columnValue = columns[j];
	            var dt = formatDate(new Date(columnValue), "MMM DD");
                var hiddenValue = "";
	            value = (value != null && value["y"]) ? hiddenValue + "â¹ " + value["y"] : "-";
	            tbodyRows = tbodyRows + "<td>" + value + "</td>";
	        }
		}
        tbodyRows = tbodyRows + "</tr>";
    }
    tbodyRows = tbodyRows + "</tbody>"

    var fullTable = headerRow + "" + tbodyRows;
    $("#historyDataTable").html(fullTable);
}

function loadDataInChart(content, chartTitle, chartDescription){
	var themeMode = getThemeMode();
    Highcharts.chart('history-chart-container', {
        chart: {
	        backgroundColor: (themeMode == "dark"?'#4b4b4b':undefined),
            height: 600,
            type: 'spline',
            description: chartDescription
        },
        title: {
            text: chartTitle,
            style: {
				color: (themeMode == "dark"?'#fff':undefined),
			}
        },
        legend: {
            maxHeight: 60,
            itemWidth: 170,
            borderWidth: 1,
			itemStyle: {
				"color": (themeMode == "dark"?'#fff':undefined),
				fontSize: '14px'
			}
        },
        xAxis: {
            categories: content.columns,
            labels: {
                style: {
				    color: (themeMode == "dark"?'#fff':undefined),
			    }
            }
        },
        yAxis: {
            title: {
                text: 'Rupees',
                style: {
				    color: (themeMode == "dark"?'#fff':undefined),
			    }
            },
            labels: {
                style: {
				    color: (themeMode == "dark"?'#fff':undefined),
			    },
                formatter: function () {
                    return 'â¹ ' + this.value;
                }
            }
        },
        tooltip: {
            shared: false,
            formatter: function () {
            	var extraData = "";
            	if(this.point.units)
        		{
            		extraData = extraData + '<tspan class="highcharts-br">&ZeroWidthSpace;</tspan>';
            		extraData = extraData + '<tspan style="fill:'+this.color+'">â</tspan> Units: '+ '<tspan style="font-weight:bold;">â¹'+this.point.units+'</tspan>';
        		}
            	if(this.point.retailprice)
        		{
            		extraData = extraData + '<tspan class="highcharts-br">&ZeroWidthSpace;</tspan>';
            		extraData = extraData + '<tspan style="fill:'+this.color+'">â</tspan> Retail Price: '+ '<tspan style="font-weight:bold;">â¹'+this.point.retailprice+'</tspan>';
        		}
            	if(this.point.shopingmallprice)
        		{
            		extraData = extraData + '<tspan class="highcharts-br">&ZeroWidthSpace;</tspan>';
            		extraData = extraData + '<tspan style="fill:'+this.color+'">â</tspan> Shopping Mall Price: '+ '<tspan style="font-weight:bold;">â¹'+this.point.shopingmallprice+'</tspan>';
        		}
            	if(this.point.onlinemallprice)
        		{
            		extraData = extraData + '<tspan class="highcharts-br">&ZeroWidthSpace;</tspan>';
            		extraData = extraData + '<tspan style="fill:'+this.color+'">â</tspan> Online/Mall Price: '+ '<tspan style="font-weight:bold;">â¹'+this.point.onlinemallprice+'</tspan>';
        		}
            	return '<text x="8" y="18" style="color:#333333;font-size:12px;fill:#333333;">'+
            	'<tspan style="font-size: 10px">'+this.key+'</tspan>'+
            	'<tspan class="highcharts-br" dy="15" x="8">&ZeroWidthSpace;</tspan>'+
            	'<tspan style="fill:'+this.color+'">â</tspan> '+this.series.name+': '+
            	'<tspan style="font-weight:bold;">â¹'+this.point.yValue+'</tspan>'+ extraData +
            	'</text>';
            }
        },
        plotOptions: {
            spline: {
                marker: {
                    radius: 4,
                    lineColor: (themeMode == "dark"?'#fff':'#666666'),
                    lineWidth: 1
                }
            }
        },
        series: content.data
    });
}

function changeHistoryPageView(view)
{
    if(view == 'table')
    {
        document.getElementById("navItemTable").classList.add("active")
        document.getElementById("navItemChart").classList.remove("active");
        document.getElementById("historyDataTableDiv").style.display = "block";
        document.getElementById("historyDataChartDiv").style.display = "none";
    }
    else
    {
        document.getElementById("navItemChart").classList.add("active")
        document.getElementById("navItemTable").classList.remove("active");
        document.getElementById("historyDataTableDiv").style.display = "none";
        document.getElementById("historyDataChartDiv").style.display = "block";
    }
}
