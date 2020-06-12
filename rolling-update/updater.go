package main

import (
	"flag"
	"fmt"
	"k8s.io/client-go/util/retry"
	"os"
	"path/filepath"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	//
	// Uncomment to load all auth plugins
	// _ "k8s.io/client-go/plugin/pkg/client/auth"
	//
	// Or uncomment to load specific auth plugins
	// _ "k8s.io/client-go/plugin/pkg/client/auth/azure"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/oidc"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/openstack"
)

const sleepingDuration = 30

func main() {
	var kubeconfig *string
	if home := homeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}
	flag.Parse()

	// use the current context in kubeconfig
	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}

	// create the clientSet
	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}
	for {
		retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// Retrieve the latest version of Deployment before attempting update
			// RetryOnConflict uses exponential backoff to avoid exhausting the apiserver
			deploymentsClient := clientSet.AppsV1().Deployments(metav1.NamespaceDefault)

			deployment, getErr := deploymentsClient.Get("deployment", metav1.GetOptions{})
			if getErr != nil {
				panic(fmt.Errorf("failed to get latest version of Deployment: %v", getErr))
			}

			deployment.Spec.Template.Spec.Containers[0].Image = "nginx:1.13" // change container version
			_, updateErr := deploymentsClient.Update(deployment) // update deployment
			return updateErr
		})
		if retryErr != nil {
			panic(fmt.Errorf("update failed: %v", retryErr))
		}
		fmt.Println("Updated container image...")

		// sleep for some time.
		time.Sleep(sleepingDuration * time.Minute)
	}
}

func homeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return os.Getenv("USERPROFILE") // windows
}
