<template>
    
    <div id='main'>
        <div id='app'>
            <NavBarComponent/>
            <div>
                <h1>Get data</h1>
                <toggle-button @change="turnSensorsOn" :value="false" color="#1DB954" :sync="true" :labels="true"/>
                {{ counter }}
            </div>
        </div>
        
    </div>
</template>


<script>
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap-vue/dist/bootstrap-vue.css';
import NavBarComponent from './NavBar.vue';

export default {
    name:'MainScreen',
    components:{NavBarComponent},
    data () {
        return{
            msg: 'Welcome to Your Vue.js App',
            counter: 0
        }
    },
    methods: {
        turnSensorsOn: function(event){
            this.$mqtt.publish('', 'message')
        }
    }
}

</script>

<style scoped>
    #main{
        background-color: #535353;
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
    }

    h1{
        color: white;
    }
</style>