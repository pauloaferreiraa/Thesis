// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import BootstrapVue from 'bootstrap-vue'
import ToggleButton from 'vue-js-toggle-button'
import VueMqtt from 'vue-mqtt'
import Msg from 'vue-message'


Vue.use(Msg, {
    text: 'Hello world', duration: 3000, background: 'rgb(30,215,96)'
})
Vue.use(VueMqtt, 'ws://iot.eclipse.org/ws'); 
Vue.use(ToggleButton)
Vue.use(BootstrapVue)

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
