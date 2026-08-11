import { Modal } from 'bootstrap';
import './sass/main.scss';
import { initialiseDepthMap } from './js/maps.js';
import { initialiseDailyRisks } from './js/risks.js';
import { initialiseRiverFlowChart } from './js/riverflow.js';

window.initialiseDepthMap = initialiseDepthMap;
window.initialiseDailyRisks = initialiseDailyRisks;
window.initialiseRiverFlowChart = initialiseRiverFlowChart;
