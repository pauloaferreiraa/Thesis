<template>
    
    <div id='main'>
        <div id='app'>
            <NavBarComponent/>
            <div>
                <h1>Get data</h1>
                <toggle-button @change="turnSensorsOn" :value="false" color="#1DB954" :sync="true" :labels="true"/>
                <div id='sensors'>
                    <span align="left">Left <status-indicator :status="sensorLeft" /></span>
                    <span align="center">Right <status-indicator :status="sensorRight" /></span>
                    <span align="right">Chest <status-indicator :status="sensorChest" /></span>
                </div>
                
            </div>
        </div>
        
    </div>
</template>


<script>
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap-vue/dist/bootstrap-vue.css';
import NavBarComponent from './NavBar.vue';
import { StatusIndicator } from 'vue-status-indicator';


export default {
    mqtt: {
    'clientLeft': function(val) {
        console.log(val.toString())
        if(val.toString() == 'on'){
            if(this.sensorLeft == 'negative'){
                this.sensorLeft = 'positive'
            }            
        }else{
            if(this.sensorLeft == 'positive'){
                this.sensorLeft = 'negative'
            } 
        }
    },
    'clientRight': function(val) {
        console.log(val.toString())
        if(val.toString() == 'on'){
            if(this.sensorRight == 'negative'){
                this.sensorRight = 'positive'
            }            
        }else{
            if(this.sensorRight == 'positive'){
                this.sensorRight = 'negative'
            } 
        }
    },
    'clientChest': function(val) {
        console.log(val.toString())
        if(val.toString() == 'on'){
            if(this.sensorChest == 'negative'){
                this.sensorChest = 'positive'
            }           
        }else{
            if(this.sensorChest == 'positive'){
                this.sensorChest = 'negative'
            } 
        }
    }    
    },
    name:'MainScreen',
    components:{NavBarComponent,StatusIndicator},
    data () {
        return{
            send: false,
            sensorLeft: "negative",
            sensorRight: "negative",
            sensorChest: "negative",
        }               
    },
    created: function(){
        this.$mqtt.subscribe('clientLeft')
        this.$mqtt.subscribe('clientRight')
        this.$mqtt.subscribe('clientChest')
        console.log('Subscribed')
    },
    methods: {
        turnSensorsOn: function(event){
            this.send = !this.send;
            if(this.send == true){
                this.$mqtt.publish('frontend', 'start')
                console.log('published')
            }else{
                 this.$mqtt.publish('frontend', 'finish')
            }
            
            this.$msg({text:'Waiting for Server\'s response', background: '#1DB954'})
            
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

    span{
        color: white;
        font-size:45px;
        margin:50px;
    }
</style>