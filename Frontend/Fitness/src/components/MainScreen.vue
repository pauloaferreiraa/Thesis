<template>
    
    <div id='main'>
        <div id='app'>
            <NavBarComponent/>
            <div>
                <h1>Get data</h1>
                <toggle-button @change="turnSensorsOn" :value="false" color="#1DB954" :sync="true" :labels="true"/>
                <h1>Left -> {{sensorLeft}}</h1>
                <h1>Right -> {{sensorRight}}</h1>
                <h1>Chest -> {{sensorChest}}</h1>
            </div>
        </div>
        
    </div>
</template>


<script>
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap-vue/dist/bootstrap-vue.css';
import NavBarComponent from './NavBar.vue';


export default {
    mqtt: {
    'client': function(val) {
      if(val.toString() == 'Left'){
          this.sensorLeft = !this.sensorLeft
          console.log(this.sensorLeft)
      }
    }
    },
    name:'MainScreen',
    components:{NavBarComponent},
    data () {
        return{
            send: false,
            sensorLeft: false,
            sensorRight: false,
            sensorChest: false,
        }               
    },
    created: function(){
        this.$mqtt.subscribe('client')
        console.log('Subscribed')
    },
    methods: {
        turnSensorsOn: function(event){
            this.send = !this.send;
            if(this.send == true){
                this.$mqtt.publish('frontend', 'start')
            }else{
                 this.$mqtt.publish('frontend', 'finish')
            }
            
            //this.$msg({text:'Waiting for Server\'s response', background: '#1DB954'})
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