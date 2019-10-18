<template>
    
    <div id='main'>
        <div id='app'>
            <NavBarComponent/>
            <div>
                <div class="body">
                    <h1 id='start'>Start <toggle-button @change="turnSensorsOn" :value="false" color="#1DB954" :sync="true" :labels="true"/></h1>
                    
                </div>               
                
                <div id='sensors' class="body">
                    <span class = 'sensors' align="center">Left Hand<status-indicator class='indicators' :status="sensorLeft" /></span>
                    <span class = 'sensors' align="center">Right Leg<status-indicator :status="sensorRight" /></span>
                    <span class = 'sensors' align="right">Chest<status-indicator :status="sensorChest" /></span>                 
                </div>

                <div class="body">
                    <div align="left" class="body">
                        <img id="temp" src="../assets/temperature.png">
                        <span class="thText" >{{temp}}ºC</span>
                    </div>
                    
                    <div align="left" class="body">
                        <img id="temp"  src="../assets/humidity.png">
                        <span class="thText">{{Math.abs(hum)}}%</span> 
                    </div>
                      
                </div>
                <div>
                    <img id="activity" src="../assets/squat.png">
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
    },
    'TempHum': function(val){
            var s = val.toString().split(' ');
            this.temp = parseInt(s[0])
            this.hum = parseInt(s[1])
            console.log(val.toString())
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
            temp: 0.0,
            hum: 0.0
        }               
    },
    created: function(){
        this.$mqtt.subscribe('clientLeft')
        this.$mqtt.subscribe('clientRight')
        this.$mqtt.subscribe('clientChest')
        this.$mqtt.subscribe('TempHum')
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

    #start{
        float: left;
        color: #05386B;
    }

    #main{
        background-color: #EEAD37;
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
    }

    span{
        color: #EDF5E1;
        font-size:45px;
        margin:50px;
        size: 5px;
    }

    h1{
        color:#EDF5E1;
    }

    .sensors{
        font-size: 20px;
    }

    #temp{

        width:5%;
    }

    #imgSensor{
        width:10%;
    }

    .thText{
        font-size: 20px;
        color: #EDF5E1;
        width: 5%;
    }

    .body{
        justify-content: space-between;
        padding: 1%;
    }
    
    #activity{
        width:20%;
    }
    
</style>

