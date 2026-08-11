import { select } from 'd3-selection';
import { scaleTime, scaleLinear } from 'd3-scale';
import { line, area, curveMonotoneX } from 'd3-shape';
import { axisBottom, axisLeft } from 'd3-axis';
import { extent, max } from 'd3-array';
import { timeDay } from 'd3-time';
import { timeFormat } from 'd3-time-format';

function parseForecastData(raw) {
  return raw.map(function (d) {
    return Object.assign({}, d, { forecast_time: new Date(d.forecast_time) });
  });
}

function renderChart(container, data, mode) {
  container.innerHTML = '';

  var margin = { top: 20, right: 30, bottom: 40, left: 60 };
  var width = container.clientWidth - margin.left - margin.right;
  var height = container.clientHeight - margin.top - margin.bottom;

  if (width <= 0 || height <= 0) {
    return;
  }

  var svg = select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr(
      'viewBox',
      '0 0 ' + (width + margin.left + margin.right) + ' ' + (height + margin.top + margin.bottom)
    )
    .append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  var x = scaleTime()
    .domain(extent(data, function (d) { return d.forecast_time; }))
    .range([0, width]);

  var yMax = mode === 'all'
    ? max(data, function (d) { return max(d.predictions); })
    : max(data, function (d) { return d.upper_centile; });

  var y = scaleLinear()
    .domain([0, yMax * 1.05])
    .nice()
    .range([height, 0]);

  svg.append('g')
    .attr('class', 'x-axis')
    .attr('transform', 'translate(0,' + height + ')')
    .call(axisBottom(x).ticks(timeDay.every(1)).tickFormat(timeFormat('%b %d')));

  svg.append('g')
    .attr('class', 'y-axis')
    .call(axisLeft(y));

  svg.append('text')
    .attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2)
    .attr('y', -45)
    .attr('text-anchor', 'middle')
    .text('River flow (m³/s)');

  if (mode === 'all') {
    var numLines = max(data, function (d) { return d.predictions.length; });
    var lineGen = line()
      .curve(curveMonotoneX)
      .defined(function (d) { return d.value !== undefined && d.value !== null; })
      .x(function (d) { return x(d.forecast_time); })
      .y(function (d) { return y(d.value); });

    for (var i = 0; i < numLines; i++) {
      var series = data.map(function (d) {
        return { forecast_time: d.forecast_time, value: d.predictions[i] };
      });
      svg.append('path')
        .datum(series)
        .attr('class', 'river-flow-line')
        .attr('fill', 'none')
        .attr('d', lineGen);
    }
  } else {
    var areaGen = area()
      .curve(curveMonotoneX)
      .x(function (d) { return x(d.forecast_time); })
      .y0(function (d) { return y(d.lower_centile); })
      .y1(function (d) { return y(d.upper_centile); });

    svg.append('path')
      .datum(data)
      .attr('class', 'river-flow-ribbon')
      .attr('d', areaGen);

    var medianLine = line()
      .curve(curveMonotoneX)
      .x(function (d) { return x(d.forecast_time); })
      .y(function (d) { return y(d.median); });

    svg.append('path')
      .datum(data)
      .attr('class', 'river-flow-median')
      .attr('fill', 'none')
      .attr('d', medianLine);
  }
}

export function initialiseRiverFlowChart(chartElementId, dataElementId, modeGroupId) {
  var dataElement = document.getElementById(dataElementId);
  var container = document.getElementById(chartElementId);
  var modeGroup = document.getElementById(modeGroupId);

  if (!dataElement || !container || !modeGroup) {
    return;
  }

  var data = parseForecastData(JSON.parse(dataElement.textContent));
  if (data.length === 0) {
    return;
  }

  function currentMode() {
    var checked = modeGroup.querySelector('input[name="river-flow-mode"]:checked');
    return checked ? checked.value : 'percentiles';
  }

  function draw() {
    renderChart(container, data, currentMode());
  }

  modeGroup.addEventListener('change', draw);
  window.addEventListener('resize', draw);

  draw();
}
