<template>
    <b-navbar toggleable="md" variant="primary"  fixed="top">

    <b-navbar-toggle target="nav_collapse"></b-navbar-toggle>

    <b-navbar-brand href="#">Fitness</b-navbar-brand>

    <b-collapse is-nav id="nav_collapse">

      <!-- Right aligned nav items -->
    <b-navbar-nav class="ml-auto">
        <b-nav-item href="#" enabled v-on:click="logout()">Logout</b-nav-item>
    </b-navbar-nav>

    </b-collapse>
    </b-navbar>
</template>

<script>
export default {
    
}
</script>

<style>

.navbar-dark{
  color: #05386B;
}

</style>